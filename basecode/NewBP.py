# new：NewBP.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from einops import rearrange, reduce
from tqdm import tqdm
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class SOAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n_channels, offset, am_compress, Psat, gss):
        batch_size, lenX = x.shape
        n_devices = lenX // n_channels
        inputs_adj = x + offset  
        # 调整后的输入信号=输入信号+固定偏置

        input_s = reduce(inputs_adj.reshape(batch_size, n_devices, n_channels),
                         'b d c -> b d', 'sum')
        # 计算每个设备的总输入功率
        # input_s 的结果维度是 (batch_size, n_devices)，代表每个批次中每个 SOA 设备所接收到的总输入功率。
        # input_s 的每个元素表示对应设备的总输入功率，即所有通道输入功率的和。

        G = torch.where(input_s <= 0,
                        torch.full_like(input_s, gss),
                        gss / (1 + input_s / Psat))
        # 当总输入功率 input_s 很小时，增益 G 接近于小信号增益 gss。
        # 随着 input_s 增大，分母变大，增益 G 随之下降。

        y = am_compress * \
            inputs_adj.reshape(batch_size, n_devices,
                               n_channels) * G.unsqueeze(-1)
        # G.unsqueeze(-1) 将 G 的维度扩展为 (batch_size, n_devices, 1)。
        # 通过 PyTorch 的广播机制，同一个设备（device）的所有通道都乘以相同的增益值 G

        # 一个设备内任何一个通道的信号变化，都会影响 input_s，从而改变整个设备的增益 G，进而影响该设备内所有其他通道的输出。
        # am_compress 是一个压缩/缩放因子，用于调整输出的幅度

        y = y.reshape(batch_size, -1)
        ctx.save_for_backward(inputs_adj, G, input_s)
        ctx.params = (n_channels, offset, am_compress, Psat, gss)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        inputs_adj, G, input_s = ctx.saved_tensors
        n_channels, offset, am_compress, Psat, gss = ctx.params
        batch_size, lenX = inputs_adj.shape
        n_devices = lenX // n_channels
        grad_output = grad_output.reshape(batch_size, n_devices, n_channels)
        inputs_adj = inputs_adj.reshape(batch_size, n_devices, n_channels)
        
        # NewBP算法：考虑串扰的梯度计算
        grad_direct = am_compress * G.unsqueeze(-1) * grad_output
        # 这部分对应于标准反向传播中的链式法则
        # 不考虑 G 随输入 inputs_adj 变化的情况（即假设 G 是常数）
        # grad_direct 计算的就是这个直接的梯度贡献

        dG_dinput_s = torch.where(input_s > 0,
                                  -gss / (Psat * (1 + input_s/Psat)**2),
                                  torch.zeros_like(input_s))
        # 这是串扰梯度的核心部分。它计算了增益 G 相对于总输入功率 input_s 的变化率。
        # 它的物理意义是：当总输入功率 input_s 变化一点点时，整个设备的增益 G 会变化多少。
        # 这个值通常是负的，因为功率增加，增益会下降（饱和效应）。

        grad_indirect = (reduce(grad_output * am_compress * inputs_adj,
                                'b d c -> b d', 'sum')
                         * dG_dinput_s).unsqueeze(-1)
        # 这个 grad_indirect 对于一个设备内的所有通道都是相同的，因为它源于共享的增益 G 的变化

        grad_total = (grad_direct + grad_indirect).reshape(batch_size, -1)
        # 相对于输入 x 的总梯度是直接梯度和间接（串扰）梯度之和
        
        return grad_total, None, None, None, None, None

class SOALayer(nn.Module):
    def __init__(self, n_channels=6, offset=15,
                 am_compress=1/2000, Psat=15, gss=200):
        super().__init__()
        self.n_channels = n_channels
        self.offset = offset
        self.am_compress = am_compress
        self.Psat = Psat
        self.gss = gss

    def forward(self, x):
        return SOAFunction.apply(x, self.n_channels, self.offset,
                                 self.am_compress, self.Psat, self.gss)

class SOANet(nn.Module):
    def __init__(self, n_channels=6):
        super().__init__()
        # 简化为两层网络：输入层->SOA层->输出层
        self.fc1 = nn.Linear(784, 60)  # 固定隐藏层神经元数为256
        self.soa = SOALayer(n_channels=n_channels)
        self.fc2 = nn.Linear(60, 10)
        self.sigmoid = nn.Sigmoid()
        
        # 使用Xavier初始化
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.soa(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def mse_loss(output, target):
    target_onehot = torch.zeros_like(output)
    target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
    return 0.5 * torch.sum((output - target_onehot)**2) / output.size(0)

# 数据预处理：仅进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

def train(model, optimizer, criterion, device):
    model.train()
    correct = 0
    total_loss = 0
    total_samples = 0
    pbar = tqdm(train_loader, desc="训练中", leave=False)
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total_samples += data.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100. * pred.eq(target).sum().item() / len(target):.2f}%"})
    avg_loss = total_loss / total_samples
    accuracy = 100. * correct / total_samples
    return avg_loss, accuracy

def test(model, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="测试中", leave=False)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target).item()
            test_loss += loss * data.size(0)
            pred = output.argmax(dim=1)
            batch_correct = pred.eq(target).sum().item()
            correct += batch_correct
            total_samples += data.size(0)
            pbar.set_postfix({"loss": f"{loss:.4f}", "acc": f"{100. * batch_correct / len(target):.2f}%"})
    avg_loss = test_loss / total_samples
    accuracy = 100. * correct / total_samples
    return avg_loss, accuracy

if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SOANet(n_channels=6).to(device)
    
    # 使用带动量的SGD优化器
    optimizer = optim.SGD(model.parameters(), lr=0.08, momentum=0.9, weight_decay=1e-4)
    
    criterion = mse_loss
    test_accuracies = []
    
    print(f"设备: {device}")
    print("开始训练 NewBP 模型...")
    
    for epoch in range(1, 51):
        train_loss, train_acc = train(model, optimizer, criterion, device)
        test_loss, test_acc = test(model, criterion, device)
        test_accuracies.append(test_acc)
        print(f"Epoch {epoch}/50 - 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%, 测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%")
    
    print(f"\n训练完成，最后一轮准确率：{test_accuracies[-1]:.2f}%")
    
    # 保存每个epoch的准确率到文件
    with open('new_bp_accuracy.txt', 'w') as f:
        for acc in test_accuracies:
            f.write(f"{acc}\n")
            
    # 保存网络规模信息
    with open('new_bp_network_info.txt', 'w', encoding='utf-8') as f:
        f.write(f"输入层神经元数: 784\n")
        f.write(f"隐藏层神经元数: {model.fc1.out_features}\n")
        f.write(f"SOA层通道数: {model.soa.n_channels}\n")
        f.write(f"输出层神经元数: {model.fc2.out_features}\n")
        f.write(f"总参数量: {sum(p.numel() for p in model.parameters())}\n")
    
    torch.save(model.state_dict(), "NewBP_model.pth")
    print("✅ 模型已保存为 NewBP_model.pth")
    
    # 将运行结果保存到数据文件
    with open('new_bp_results.txt', 'w') as f:
        f.write(f"{model.fc1.out_features},{model.soa.n_channels},{test_accuracies[-1]:.4f}\n")
    print("✅ 运行结果已保存到数据文件")

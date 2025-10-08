# new：fNewBP.py
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
        input_s = reduce(inputs_adj.reshape(batch_size, n_devices, n_channels),
                         'b d c -> b d', 'sum')
        G = torch.where(input_s <= 0,
                        torch.full_like(input_s, gss),
                        gss / (1 + input_s / Psat))
        y = am_compress * \
            inputs_adj.reshape(batch_size, n_devices,
                               n_channels) * G.unsqueeze(-1)
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
        dG_dinput_s = torch.where(input_s > 0,
                                  -gss / (Psat * (1 + input_s/Psat)**2),
                                  torch.zeros_like(input_s))
        grad_indirect = (reduce(grad_output * am_compress * inputs_adj,
                                'b d c -> b d', 'sum')
                         * dG_dinput_s).unsqueeze(-1)
        grad_total = (grad_direct + grad_indirect).reshape(batch_size, -1)
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

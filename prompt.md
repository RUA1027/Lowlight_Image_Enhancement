# **任务1：构建可配置的 NewBP-Net 核心模块**

**角色**: 你是一名资深的 PyTorch 算法工程师，精通自定义网络层和梯度反向传播，代码风格严谨且模块化。

**背景**: 我们正在构建一个名为 `NewBP-Net` 的图像去噪模型。其核心在于一个自定义的 `NewBPLayer`，该层用于模拟图像传感器中的物理串扰效应。根据我们的研究方案，这个层必须支持两种串扰模式：
1.  **`panchromatic` (全色模式)**: 对输入的R, G, B三个通道施加**完全相同**的串扰卷积核。
2.  **`rgb` (分色模式)**: 对R, G, B三个通道**分别**施加不同的串扰卷积核，以模拟串扰的波长依赖性。

这个模块的设计必须是灵活的，以便在后续实验中通过参数轻松切换模式。

**指令**:

### **第一部分: 实现核心 `NewBPLayer`**

1.  **创建文件**: 创建一个名为 `newbp_layer.py` 的新文件。

2.  **编写 `NewBPFunction`**: 复制以下代码到 `newbp_layer.py` 中。这是用于重载反向传播逻辑的 `autograd.Function`，其前向传播为标准卷积，反向传播为转置卷积。

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    class NewBPFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, kernel, padding, groups):
            ctx.save_for_backward(kernel)
            ctx.padding = padding
            ctx.groups = groups
            return F.conv2d(input, kernel, padding=padding, groups=groups)

        @staticmethod
        def backward(ctx, grad_output):
            kernel, = ctx.saved_tensors
            padding = ctx.padding
            groups = ctx.groups
            grad_input = F.conv_transpose2d(grad_output, kernel, padding=padding, groups=groups)
            return grad_input, None, None, None
    ```

3.  **实现可配置的 `NewBPLayer` 类**: 在 `NewBPFunction` 下方，实现 `NewBPLayer` 类。它必须严格遵循以下设计：

    * `__init__` 方法的签名为 `def __init__(self, in_channels=3, kernel_type='panchromatic', kernel_spec='P2'):`。
    * 根据 `kernel_spec` 和 `kernel_type` 参数，从以下预设值中精确地加载核数值。我们使用“中等(moderate)”强度版本作为主对比核。
        * **如果 `kernel_spec` 是 'P2'**:
            * 全色核矩阵:
                ```
                [[0.0100, 0.0200, 0.0100],
                 [0.0200, 0.8800, 0.0200],
                 [0.0100, 0.0200, 0.0100]]
                ```
        * **如果 `kernel_spec` 是 'B2'**:
            * R 核矩阵:
                ```
                [[0.0117, 0.0233, 0.0117],
                 [0.0233, 0.8600, 0.0233],
                 [0.0117, 0.0233, 0.0117]]
                ```
            * G 核矩阵:
                ```
                [[0.0100, 0.0200, 0.0100],
                 [0.0200, 0.8800, 0.0200],
                 [0.0100, 0.0200, 0.0100]]
                ```
            * B 核矩阵:
                ```
                [[0.0083, 0.0167, 0.0083],
                 [0.0167, 0.9000, 0.0167],
                 [0.0083, 0.0167, 0.0083]]
                ```
    * 根据 `kernel_type` 来构建最终的 `self.kernel` 张量：
        * **如果 `kernel_type` 是 `'panchromatic'`**: 将 'P2' 核转换为 `(1, 1, 3, 3)` 的张量，然后使用 `.repeat(in_channels, 1, 1, 1)` 将其复制成 `(3, 1, 3, 3)` 的形状。
        * **如果 `kernel_type` 是 `'rgb'`**: 分别将 R, G, B 三个核转换为 `(1, 1, 3, 3)` 的张量，然后使用 `torch.cat` 在第0维（out_channels）上将它们堆叠成一个 `(3, 1, 3, 3)` 的张量。
    * 将构建好的 `self.kernel` 封装为 `nn.Parameter`，并**严格设置 `requires_grad=False`**。
    * `forward` 方法的实现必须简洁：
        * 计算所需的 padding: `padding = (self.kernel.shape[-1] - 1) // 2`。
        * 直接调用 `NewBPFunction.apply(x, self.kernel, padding, self.in_channels)`。利用 `groups=in_channels` 的特性，可以统一处理两种模式，无需 `if/else` 判断。

### **第二部分: 构建 `NewBP-Net` 模型架构**

1.  **创建文件**: 创建一个名为 `newbp_net_arch.py` 的新文件。

2.  **编写工厂函数**: 在该文件中，实现一个名为 `create_newbp_net` 的工厂函数，用于自动化构建 `NewBP-Net`。
    * 函数签名为 `def create_newbp_net(in_channels=3, kernel_type='panchromatic', kernel_spec='P2', nafnet_params={}):`。
    * 在该函数内部：
        1.  导入原始 `NAFNet` 架构。假设其路径为 `NAFNet.basicsr.models.archs.nafnet_arch`。
        2.  导入你刚刚创建的 `NewBPLayer`。
        3.  使用 `nafnet_params` 字典实例化一个原始的 `NAFNet` 模型。
        4.  实例化一个配置好的 `NewBPLayer(in_channels=in_channels, kernel_type=kernel_type, kernel_spec=kernel_spec)`。
        5.  执行“手术”：用 `NewBPLayer` 实例替换掉 `NAFNet` 实例的 `intro` 模块（根据NAFNet源码，`model.intro` 即为第一个卷积层）。
        6.  打印一条成功信息，说明已成功创建了何种配置的 `NewBP-Net`。
        7.  返回改造后的模型实例。

至于NewBP算法的具体实现和代码细节,你可以参考文件@fNewBP.py ,这个文件中详细描述了NewBP的实现过程与原理.而对固定的串扰卷积核的数学推理和实际情况赋值,你可以参考@串扰卷积核推导.md @串扰核的“数值微调”方案（让算法优势更可见）.md 这两个文件内容.

除此之外,请你严格按照我们实验的计划步骤,始终以核心目的为向导进行编程工作,并且遵守我们规定的编程范式.

请开始你的工作!
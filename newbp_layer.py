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


class NewBPLayer(nn.Module):
    def __init__(self, in_channels=3, kernel_type='panchromatic', kernel_spec='P2'):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_type = kernel_type
        self.kernel_spec = kernel_spec

        if kernel_spec not in {'P2', 'B2'}:
        # P2代指全色核, B2代指分色核
            raise ValueError(f"Unsupported kernel_spec '{kernel_spec}'. Expected 'P2' or 'B2'.")

        if kernel_type not in {'panchromatic', 'rgb'}:
            raise ValueError(f"Unsupported kernel_type '{kernel_type}'. Expected 'panchromatic' or 'rgb'.")

        if kernel_type == 'rgb' and in_channels != 3:
            raise ValueError("kernel_type 'rgb' requires in_channels to be 3.")

        self.kernel = nn.Parameter(self._build_kernel(), requires_grad=False)

    def _build_kernel(self):
        p2_kernel = torch.tensor(
            [[0.0100, 0.0200, 0.0100],
             [0.0200, 0.8800, 0.0200],
             [0.0100, 0.0200, 0.0100]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        # view（输出通道数，输入通道数，卷积核高度，卷积核宽度）

        if self.kernel_type == 'panchromatic':
            if self.kernel_spec != 'P2':
                raise ValueError("kernel_type 'panchromatic' requires kernel_spec 'P2'.")
            return p2_kernel.repeat(self.in_channels, 1, 1, 1)

        if self.kernel_spec == 'P2':
            raise ValueError("kernel_type 'rgb' requires kernel_spec 'B2'.")

        red_kernel = torch.tensor(
            [[0.0117, 0.0233, 0.0117],
             [0.0233, 0.8600, 0.0233],
             [0.0117, 0.0233, 0.0117]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        green_kernel = torch.tensor(
            [[0.0100, 0.0200, 0.0100],
             [0.0200, 0.8800, 0.0200],
             [0.0100, 0.0200, 0.0100]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        blue_kernel = torch.tensor(
            [[0.0083, 0.0167, 0.0083],
             [0.0167, 0.9000, 0.0167],
             [0.0083, 0.0167, 0.0083]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        return torch.cat((red_kernel, green_kernel, blue_kernel), dim=0)

    def forward(self, x):
        padding = (self.kernel.shape[-1] - 1) // 2
        return NewBPFunction.apply(x, self.kernel, padding, self.in_channels)

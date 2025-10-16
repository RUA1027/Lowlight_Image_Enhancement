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
    def backward(ctx, grad_outputs):
        kernel, = ctx.saved_tensors
        padding = ctx.padding
        groups = ctx.groups
        grad_input = F.conv_transpose2d(grad_outputs, kernel, padding=padding, groups=groups)
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


class CrosstalkPSF(nn.Module):
    """
    Fixed PSF used ONLY in the loss graph (output-side consistency).

    - mode='mono': kernels shape [1,1,3,3], broadcast depthwise to C channels
    - mode='rgb' : kernels shape [3,1,3,3], requires input C==3 (sRGB)
    """
    def __init__(self, mode: str, kernels: torch.Tensor):
        super().__init__()
        assert mode in {"mono", "rgb"}
        self.mode = mode
        # K is a state (buffer), not a parameter. It is saved/moved with the model
        # but will not be optimized. Ensure energy preservation (sum≈1 per channel).
        self.register_buffer("kernel", kernels.clone(), persistent=True)
        with torch.no_grad():
            k = self.kernel
            # Normalize per-kernel so that sum of weights ≈ 1
            s = k.view(k.shape[0], -1).sum(dim=1, keepdim=True).clamp_min(1e-12)
            self.kernel = (k / s.view(-1, 1, 1, 1))
        self.kernel: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward is ONLY used in the loss graph.
        Invariants (Scenario B):
        - Forward of backbone remains identity w.r.t. K (no input-side conv).
        - Here we apply K depthwise with groups=3 on sRGB tensors.
        - padding=1, stride=1 to keep spatial size.
        """
        C = x.shape[1]
        assert C == 3, "CrosstalkPSF expects sRGB inputs (3 channels)."
        k = self.kernel
        if self.mode == "mono":
            assert k.shape == (1, 1, 3, 3), "mono mode expects kernels of shape [1,1,3,3]"
            k = k.expand(3, 1, 3, 3)  # broadcast to 3 channels
        else:
            assert k.shape == (3, 1, 3, 3), "rgb mode expects kernels of shape [3,1,3,3]"
        # depthwise conv across RGB channels (groups=3)
        return F.conv2d(x, k, bias=None, stride=1, padding=1, groups=3)


def build_psf_kernels(mode: str, kernel_spec: str = 'P2') -> torch.Tensor:
    """
    Build canonical PSF kernels from existing specs used by NewBPLayer.

    - mode='mono' & kernel_spec='P2' => [1,1,3,3]
    - mode='rgb'  & kernel_spec='B2' => [3,1,3,3]
    """
    if mode not in {"mono", "rgb"}:
        raise ValueError("mode must be 'mono' or 'rgb'")

    # Base kernels reused from NewBPLayer._build_kernel
    p2_kernel = torch.tensor(
        [[0.0100, 0.0200, 0.0100],
         [0.0200, 0.8800, 0.0200],
         [0.0100, 0.0200, 0.0100]],
        dtype=torch.float32
    ).view(1, 1, 3, 3)

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

    if mode == 'mono':
        if kernel_spec != 'P2':
            raise ValueError("mono mode expects kernel_spec 'P2'")
        return p2_kernel  # [1,1,3,3]
    else:
        if kernel_spec != 'B2':
            raise ValueError("rgb mode expects kernel_spec 'B2'")
        return torch.cat((red_kernel, green_kernel, blue_kernel), dim=0)  # [3,1,3,3]

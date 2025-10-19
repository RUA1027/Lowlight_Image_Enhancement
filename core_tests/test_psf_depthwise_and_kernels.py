"""
Test objectives:
1. Depthwise semantics: groups == in_channels for per-channel convolution, mono kernels [1,1,3,3] expand to [3,1,3,3], and rgb kernels [3,1,3,3] match manual per-channel conv2d results.
2. Same-size outputs: 3x3 kernels with stride=1 and padding=1 must preserve spatial dimensions.
3. Energy preservation & non-negativity: each PSF kernel sums to ≈1 and contains no negative weights.
4. Kernel stored as buffer: K is registered via register_buffer and must not appear in parameters().
"""

from __future__ import annotations

import os
import sys

# 将项目根目录和 NAFNet 目录添加到 Python 路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
nafnet_root = os.path.join(project_root, 'NAFNet')
for path in [project_root, nafnet_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

import pytest
import torch
import torch.nn.functional as F

from NewBP_model.newbp_layer import CrosstalkPSF


def _rand_img(n: int = 1, c: int = 3, h: int = 32, w: int = 32, *, dtype: torch.dtype = torch.float32, device: str = "cpu") -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(n, c, h, w, dtype=dtype, device=device)


def test_mono_broadcast_depthwise_equivalence_cpu() -> None:
    """
    Mono kernels must broadcast to three channels and match manual depthwise results.
    """
    x = _rand_img(1, 3, 31, 29)
    k_mono = torch.tensor(
        [[[[0.0, 0.05, 0.0], [0.05, 0.80, 0.05], [0.0, 0.05, 0.0]]]],
        dtype=torch.float32,
    )
    psf = CrosstalkPSF(mode="mono", kernels=k_mono)

    y = psf(x)
    k_depthwise = psf.kernel.expand(3, -1, -1, -1)
    y_ref = F.conv2d(x, k_depthwise, padding=1, groups=3)

    assert y.shape == x.shape, "3x3 kernels with stride=1 and padding=1 should keep spatial size unchanged."
    assert torch.allclose(y, y_ref, atol=1e-6), "Mono PSF must broadcast and behave identically to depthwise conv."


@pytest.mark.parametrize(
    "k_r,k_g,k_b",
    [
        (
            torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32),
            torch.tensor([[[[0, 1, 0], [1, 2, 1], [0, 1, 0]]]], dtype=torch.float32) / 6.0,
            torch.tensor([[[[1, 0, 1], [0, 1, 0], [1, 0, 1]]]], dtype=torch.float32) / 5.0,
        ),
    ],
)
def test_rgb_per_channel_depthwise_equivalence_cpu(
    k_r: torch.Tensor, k_g: torch.Tensor, k_b: torch.Tensor
) -> None:
    """
    RGB kernels must operate depthwise, matching manual per-channel convolution results.
    """
    x = _rand_img(2, 3, 28, 28)
    k_rgb = torch.cat([k_r, k_g, k_b], dim=0)
    psf = CrosstalkPSF(mode="rgb", kernels=k_rgb)

    y = psf(x)

    y_r = F.conv2d(x[:, 0:1], psf.kernel[0:1], padding=1)
    y_g = F.conv2d(x[:, 1:2], psf.kernel[1:2], padding=1)
    y_b = F.conv2d(x[:, 2:3], psf.kernel[2:3], padding=1)
    y_ref = torch.cat([y_r, y_g, y_b], dim=1)

    assert y.shape == x.shape
    assert torch.allclose(y, y_ref, atol=1e-6), "RGB PSF must equal manual per-channel convolution."


def test_kernel_physical_constraints_and_same_size() -> None:
    """
    PSF kernels should be normalized to sum≈1, be non-negative, and preserve spatial size.
    """
    k_raw = torch.tensor(
        [[[[0.0, 1.0, 0.0], [1.0, 6.0, 1.0], [0.0, 1.0, 0.0]]]],
        dtype=torch.float32,
    )
    psf = CrosstalkPSF("mono", k_raw.clone())

    k_inner = psf.kernel
    sums = k_inner.flatten(1).sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), "PSF kernels should preserve energy (sum≈1)."
    assert torch.all(k_inner >= 0), "PSF kernels should be non-negative (intensity/probability interpretation)."

    x = _rand_img(1, 3, 21, 17)
    y = psf(x)
    assert y.shape == x.shape, "3x3 kernels with unit stride/padding must keep feature map size."


def test_kernel_is_buffer_not_parameter() -> None:
    """
    Kernels must be registered as buffers (no trainable parameters) yet appear in state_dict.
    """
    k = torch.ones(1, 1, 3, 3, dtype=torch.float32) / 9.0
    psf = CrosstalkPSF("mono", k)

    param_count = sum(p.numel() for p in psf.parameters())
    assert param_count == 0, "PSF kernels should not be exposed as trainable parameters."

    state_keys = psf.state_dict().keys()
    assert any("kernel" in name for name in state_keys), "PSF kernels should persist as buffers in state_dict."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_mono_equivalence_amp_smoke() -> None:
    """
    CUDA/AMP smoke test to ensure mono kernels behave depthwise on GPU with mixed precision.
    """
    x = _rand_img(1, 3, 40, 40, device="cuda")
    k = torch.tensor(
        [[[[0.0, 0.1, 0.0], [0.1, 0.6, 0.1], [0.0, 0.1, 0.0]]]],
        dtype=torch.float32,
        device="cuda",
    )
    psf = CrosstalkPSF("mono", k)

    with torch.cuda.amp.autocast():
        y = psf(x)
        y_ref = F.conv2d(x, psf.kernel.expand(3, -1, -1, -1), padding=1, groups=3)

    assert y.shape == x.shape
    assert torch.allclose(y, y_ref, atol=1e-5), "Mono PSF should match depthwise convolution under AMP."


# Remarks:
# Depthwise (groups=in_channels) conv keeps the backbone output aligned while only applying PSF in the loss path.
# PSF kernels represent optical energy distributions, hence the focus on normalization and non-negativity.

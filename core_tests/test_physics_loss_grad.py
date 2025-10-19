"""
Theory anchor: ConvTranspose2d is the input-gradient operator of Conv2d, so placing K only in the loss path naturally yields K^T during backprop; no input-side convolution is required.
Numerical strategy: gradcheck demands double precision, requires_grad tensors, and avoidance of non-differentiable points (L1 at zero). Add small perturbations so residuals stay away from zero before checking subgradients.
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
from torch.autograd import gradcheck

from NewBP_model.newbp_layer import CrosstalkPSF


def _make_inputs(
    *,
    batch: int = 1,
    channels: int = 3,
    height: int = 8,
    width: int = 8,
    eps: float = 1e-3,
    dtype: torch.dtype = torch.double,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    bhat = torch.randn(batch, channels, height, width, dtype=dtype, requires_grad=True)
    target = torch.randn(batch, channels, height, width, dtype=dtype)
    # Apply a small perturbation so residuals rarely sit exactly at zero (L1 subgradient ambiguity).
    target = target + eps * torch.randn_like(target)
    return bhat, target


def _prepare_kernel(mode: str, dtype: torch.dtype) -> torch.Tensor:
    if mode == "mono":
        base = torch.tensor(
            [[[[0.0, 0.1, 0.0], [0.1, 0.6, 0.1], [0.0, 0.1, 0.0]]]],
            dtype=dtype,
        )
        return base
    if mode == "rgb":
        k_r = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]], dtype=dtype)
        k_g = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]]], dtype=dtype) / 6.0
        k_b = torch.tensor([[[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]]], dtype=dtype) / 5.0
        return torch.cat([k_r, k_g, k_b], dim=0)
    raise ValueError(f"Unsupported mode: {mode}")


def _depthwise_kernel(psf: CrosstalkPSF) -> torch.Tensor:
    if psf.mode == "mono":
        return psf.kernel.expand(3, -1, -1, -1).contiguous()
    return psf.kernel.contiguous()


@pytest.mark.parametrize("mode", ["mono", "rgb"])
def test_adjoint_matches_conv_transpose(mode: str) -> None:
    """
    grad_Bhat <gA, K*Bhat> must equal conv_transpose2d(gA, K) for both mono and rgb kernels.
    """
    dtype = torch.double
    bhat, _ = _make_inputs(dtype=dtype)
    kernels = _prepare_kernel(mode, dtype)
    psf = CrosstalkPSF(mode, kernels).to(dtype=dtype)

    ahat = psf(bhat)
    gA = torch.randn_like(ahat)
    ahat.backward(gA)
    grad_auto = bhat.grad.detach().clone()
    bhat.grad.zero_()

    kernel_dw = _depthwise_kernel(psf)
    grad_adj = F.conv_transpose2d(gA, kernel_dw, padding=1, groups=3)
    assert torch.allclose(
        grad_auto,
        grad_adj,
        atol=1e-6,
    ), "Autograd gradient must match the conv_transpose2d adjoint (K^T)."


@pytest.mark.parametrize("mode", ["mono", "rgb"])
def test_l1_analytic_grad_matches_autograd(mode: str) -> None:
    """
    L1 gradient should equal K^T * sign(residual) once residuals avoid the non-differentiable origin.
    """
    dtype = torch.double
    bhat, align = _make_inputs(dtype=dtype)
    kernels = _prepare_kernel(mode, dtype)
    psf = CrosstalkPSF(mode, kernels).to(dtype=dtype)

    kernel_dw = _depthwise_kernel(psf)
    ahat = F.conv2d(bhat, kernel_dw, padding=1, groups=3)
    residual = ahat - align

    loss = residual.abs().sum()
    loss.backward()
    grad_auto = bhat.grad.detach().clone()
    bhat.grad.zero_()

    grad_analytic = F.conv_transpose2d(torch.sign(residual).detach(), kernel_dw, padding=1, groups=3)
    assert torch.allclose(
        grad_auto,
        grad_analytic,
        atol=1e-6,
    ), "Analytic L1 gradient K^T * sign(residual) must match Autograd."


@pytest.mark.parametrize("mode", ["mono", "rgb"])
def test_gradcheck_physics_loss(mode: str) -> None:
    """
    gradcheck must succeed for f(z) = ||K*z - A||_1 under mono and rgb kernels.
    """
    dtype = torch.double
    torch.manual_seed(0)
    z = torch.randn(1, 3, 6, 6, dtype=dtype, requires_grad=True)
    if mode == "mono":
        k = torch.ones(1, 1, 3, 3, dtype=dtype) / 9.0
    else:
        k = torch.ones(3, 1, 3, 3, dtype=dtype) / 9.0
    psf = CrosstalkPSF(mode, k).to(dtype=dtype)

    reference = torch.randn_like(z).detach() + 1e-3 * torch.randn_like(z)

    def func(inp: torch.Tensor) -> torch.Tensor:
        return (psf(inp) - reference).abs().sum()

    assert gradcheck(func, (z,), eps=1e-6, atol=1e-4), "gradcheck failed; ensure double precision and smooth residuals."


# Remarks:
# For additional robustness one can switch to torch.autograd.functional.vjp / torch.func.vjp to verify VJP stability.
# Depthwise semantics (groups == in_channels) are defined in PyTorch Conv2d documentation and remain intact in these tests.

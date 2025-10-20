"""
Linear-domain PSNR/SSIM regression tests against unified evaluation guidelines.

References for expected behaviour:
- PSNR definition: 10 * log10(MAX_I^2 / MSE); identical images yield infinite PSNR.
  (Wikipedia: Peak signal-to-noise ratio)
- SSIM definition and default parameters: Wang et al., 2004; Gaussian window with
  kernel_size=11, sigma≈1.5, k1=0.01, k2=0.03; kernel_size must be odd.
- SSIM usage guidance: TorchMetrics / scikit-image documentation emphasises correct
  data_range for float tensors and spatial averaging consistency.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../NAFNet")))

import math
from typing import Callable, Literal

import pytest
import torch
import torch.nn.functional as F

from metrics.linear import psnr_linear, ssim_linear


@pytest.fixture(scope="module", params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request: pytest.FixtureRequest) -> torch.device:
    return torch.device(request.param)


@pytest.fixture()
def make_images(device: torch.device) -> Callable[..., torch.Tensor]:
    def _factory(
        n: int = 2,
        c: int = 3,
        h: int = 48,
        w: int = 48,
        kind: str = "01",
    ) -> torch.Tensor:
        x = torch.rand(n, c, h, w, device=device)
        if kind == "255":
            x = x * 255.0
        elif kind == "neg1":
            x = x * 2.0 - 1.0
        return x.to(dtype=torch.float32)

    return _factory


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


@pytest.mark.parametrize("ndim", [3, 4])
def test_psnr_identity_returns_infinite(device: torch.device, make_images: Callable[..., torch.Tensor], ndim: int) -> None:
    torch.manual_seed(0)
    data = make_images(n=1, c=3, h=32, w=32)
    if ndim == 3:
        pred = data[0]
        target = data[0].clone()
    else:
        pred = data.clone()
        target = data.clone()

    psnr_val = psnr_linear(pred, target, data_range=1.0, reduction="mean")
    assert math.isinf(_to_float(psnr_val))


def test_psnr_data_range_equivalence(device: torch.device, make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(1)
    gt01 = make_images(n=2, c=3, h=40, w=40, kind="01")
    noise = 0.02 * torch.randn_like(gt01)
    pred01 = (gt01 + noise).clamp(0.0, 1.0)

    gt255 = gt01 * 255.0
    pred255 = pred01 * 255.0

    gt_neg = gt01 * 2.0 - 1.0
    pred_neg = pred01 * 2.0 - 1.0

    p01 = psnr_linear(gt01, pred01, data_range=1.0, reduction="none")
    p255 = psnr_linear(gt255, pred255, data_range=255.0, reduction="none")
    pneg = psnr_linear(gt_neg, pred_neg, data_range=2.0, reduction="none")

    assert torch.allclose(p01.cpu(), p255.cpu(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(p01.cpu(), pneg.cpu(), atol=1e-5, rtol=1e-5)


def test_psnr_reduction_modes(device: torch.device, make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(2)
    gt = make_images(n=4, c=3, h=32, w=32)
    pred = (gt + 0.03 * torch.randn_like(gt)).clamp(0.0, 1.0)

    psnr_none = psnr_linear(gt, pred, data_range=1.0, reduction="none")
    psnr_mean = psnr_linear(gt, pred, data_range=1.0, reduction="mean")
    psnr_sum = psnr_linear(gt, pred, data_range=1.0, reduction="sum")

    assert psnr_none.shape == (gt.shape[0],)
    assert abs(psnr_none.mean().item() - _to_float(psnr_mean)) <= 1e-6
    assert abs(psnr_none.sum().item() - _to_float(psnr_sum)) <= 1e-5


def test_psnr_monotonic_with_noise(device: torch.device, make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(3)
    gt = make_images(n=1, c=3, h=48, w=48)
    noise_levels = [0.0, 0.01, 0.05]
    scores = []
    for sigma in noise_levels:
        pred = (gt + sigma * torch.randn_like(gt)).clamp(0.0, 1.0)
        val = psnr_linear(gt, pred, data_range=1.0)
        scores.append(_to_float(val))

    for prev, nxt in zip(scores, scores[1:]):
        assert nxt <= prev + 1e-4


def test_psnr_invalid_arguments(device: torch.device) -> None:
    torch.manual_seed(4)
    gt = torch.rand(1, 3, 16, 16, device=device, dtype=torch.float32)
    pred = torch.rand(1, 3, 16, 16, device=device, dtype=torch.float32)

    with pytest.raises(ValueError):
        psnr_linear(gt, pred, data_range=-1.0)
    with pytest.raises(ValueError):
        psnr_linear(gt[..., :14], pred, data_range=1.0)

    ints = torch.randint(0, 10, (3, 16, 16), device=device, dtype=torch.int64)
    with pytest.raises(TypeError):
        psnr_linear(ints, ints.clone(), data_range=10.0)


def test_ssim_identity_close_to_one(device: torch.device, make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(5)
    gt = make_images(n=2, c=3, h=64, w=64)
    val = ssim_linear(
        gt,
        gt.clone(),
        data_range=1.0,
        kernel_size=11,
        sigma=1.5,
        k1=0.01,
        k2=0.03,
        gaussian=True,
        channel_aggregate="mean",
        padding="reflect",
    )
    assert math.isclose(_to_float(val), 1.0, rel_tol=1e-6, abs_tol=1e-6)


@pytest.mark.parametrize("kernel_size", [3, 5, 11, 15])
def test_ssim_kernel_size_validation(device: torch.device, make_images: Callable[..., torch.Tensor], kernel_size: int) -> None:
    torch.manual_seed(6)
    gt = make_images(n=1, c=3, h=32, w=32)
    pred = (gt + 0.01 * torch.randn_like(gt)).clamp(0.0, 1.0)

    if kernel_size % 2 == 0 or kernel_size > 32:
        with pytest.raises(ValueError):
            ssim_linear(gt, pred, data_range=1.0, kernel_size=kernel_size)
    else:
        ssim_linear(gt, pred, data_range=1.0, kernel_size=kernel_size)


def test_ssim_requires_valid_spatial_extent(device: torch.device, make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(7)
    gt = make_images(n=1, c=3, h=10, w=10)
    pred = gt.clone()
    with pytest.raises(ValueError):
        ssim_linear(gt, pred, data_range=1.0, kernel_size=11)


def test_ssim_data_range_equivalence(device: torch.device, make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(8)
    gt01 = make_images(n=2, c=3, h=40, w=40, kind="01")
    pred01 = (gt01 + 0.015 * torch.randn_like(gt01)).clamp(0.0, 1.0)

    gt255 = gt01 * 255.0
    pred255 = pred01 * 255.0
    gt_neg = gt01 * 2.0 - 1.0
    pred_neg = pred01 * 2.0 - 1.0

    s01 = ssim_linear(gt01, pred01, data_range=1.0, kernel_size=11, sigma=1.5)
    s255 = ssim_linear(gt255, pred255, data_range=255.0, kernel_size=11, sigma=1.5)
    sneg = ssim_linear(gt_neg, pred_neg, data_range=2.0, kernel_size=11, sigma=1.5)

    assert math.isclose(_to_float(s01), _to_float(s255), rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(_to_float(s01), _to_float(sneg), rel_tol=1e-5, abs_tol=1e-5)


@pytest.mark.parametrize("padding_mode", ["reflect", "replicate", "circular", "constant"])
def test_ssim_supports_padding_modes(device: torch.device, make_images: Callable[..., torch.Tensor], padding_mode: str) -> None:
    torch.manual_seed(9)
    gt = make_images(n=1, c=3, h=32, w=32)
    pred = (gt + 0.01 * torch.randn_like(gt)).clamp(0.0, 1.0)
    padding: Literal["reflect", "replicate", "circular", "constant"] = padding_mode  # type: ignore[assignment]
    val = ssim_linear(
        gt,
        pred,
        data_range=1.0,
        padding=padding,
        kernel_size=11,
        sigma=1.5,
    )
    assert torch.isfinite(val.detach()).all()


def test_ssim_channel_aggregate(device: torch.device, make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(10)
    gt = make_images(n=1, c=3, h=48, w=48)
    pred = (gt + 0.02 * torch.randn_like(gt)).clamp(0.0, 1.0)

    s_mean = ssim_linear(gt, pred, data_range=1.0, channel_aggregate="mean", reduction="none")
    s_none = ssim_linear(gt, pred, data_range=1.0, channel_aggregate="none", reduction="none")

    assert s_none.shape == (gt.shape[0], gt.shape[1])
    mean_from_none = s_none.mean(dim=1)
    assert torch.allclose(s_mean.view(-1), mean_from_none.view(-1), atol=1e-6, rtol=1e-6)


def test_ssim_reduction_modes(device: torch.device, make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(11)
    gt = make_images(n=3, c=3, h=36, w=36)
    pred = (gt + 0.025 * torch.randn_like(gt)).clamp(0.0, 1.0)

    s_none = ssim_linear(gt, pred, data_range=1.0, reduction="none", channel_aggregate="mean")
    s_mean = ssim_linear(gt, pred, data_range=1.0, reduction="mean", channel_aggregate="mean")
    s_sum = ssim_linear(gt, pred, data_range=1.0, reduction="sum", channel_aggregate="mean")

    assert s_none.shape == (gt.shape[0],)
    assert torch.allclose(s_none.mean(), s_mean, atol=1e-6, rtol=1e-6)
    assert torch.allclose(s_none.sum(), s_sum, atol=1e-6, rtol=1e-6)


def test_ssim_monotonic_with_degradation(device: torch.device, make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(12)
    gt = make_images(n=1, c=3, h=64, w=64)

    blurred = F.avg_pool2d(gt, kernel_size=5, stride=1, padding=2)
    noisy = (gt + 0.05 * torch.randn_like(gt)).clamp(0.0, 1.0)

    s_gt = ssim_linear(gt, gt.clone(), data_range=1.0)
    s_blur = ssim_linear(gt, blurred, data_range=1.0)
    s_noise = ssim_linear(gt, noisy, data_range=1.0)

    ref = _to_float(s_gt)
    blur_score = _to_float(s_blur)
    noise_score = _to_float(s_noise)
    assert blur_score <= ref + 1e-4
    assert noise_score <= blur_score + 1e-4


def test_ssim_invalid_arguments(device: torch.device, make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(13)
    gt = make_images(n=1, c=3, h=32, w=32)
    pred = make_images(n=1, c=3, h=32, w=32)

    with pytest.raises(ValueError):
        ssim_linear(gt, pred, data_range=0.0)
    with pytest.raises(ValueError):
        ssim_linear(gt, pred, data_range=1.0, kernel_size=10)
    with pytest.raises(ValueError):
        ssim_linear(gt, pred, data_range=1.0, k1=-0.1)
    with pytest.raises(ValueError):
        ssim_linear(gt, pred, data_range=1.0, channel_aggregate="median")  # type: ignore[arg-type]

    ints = torch.randint(0, 10, (3, 32, 32), device=device, dtype=torch.int64)
    with pytest.raises(TypeError):
        ssim_linear(ints, ints.clone(), data_range=10.0)

"""
Channel-wise RGB metrics regression suite (PSNR, CPSNR, SSIM).

This suite enforces the definitions used in demosaicking/restoration literature:
- Per-channel PSNR averaged across RGB, and CPSNR computed from the joint RGB MSE.
- Structural similarity (SSIM) with Wang et al. (2004) parameters and correct data_range handling.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../NAFNet")))

import math
from typing import Callable, Dict, Literal

import pytest
import torch

from metrics.channelwise import cpsnr_rgb, rgb_psnr, rgb_ssim


@pytest.fixture(scope="module", params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request: pytest.FixtureRequest) -> torch.device:
    return torch.device(request.param)


@pytest.fixture()
def make_rgb(device: torch.device) -> Callable[..., torch.Tensor]:
    def _factory(
        n: int = 2,
        h: int = 32,
        w: int = 32,
        kind: str = "01",
    ) -> torch.Tensor:
        data = torch.rand(n, 3, h, w, device=device)
        if kind == "255":
            data = (data * 255.0).round()
        elif kind == "neg1":
            data = data * 2.0 - 1.0
        return data.to(dtype=torch.float32)

    return _factory


def _as_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def test_requires_three_channels(device: torch.device) -> None:
    torch.manual_seed(0)
    pred = torch.rand(1, 1, 8, 8, device=device)
    target = torch.rand_like(pred)
    with pytest.raises(ValueError):
        rgb_psnr(pred, target)
    with pytest.raises(ValueError):
        cpsnr_rgb(pred, target)
    with pytest.raises(ValueError):
        rgb_ssim(pred, target)


def test_accepts_3d_tensors(make_rgb: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(1)
    sample = make_rgb(n=1)[0]
    result = rgb_psnr(sample, sample, data_range=1.0)
    assert {"R", "G", "B", "mean"} <= result.keys()
    ssim = rgb_ssim(sample, sample, data_range=1.0, channel_aggregate="none")
    assert {"R", "G", "B", "mean"} <= ssim.keys()


@pytest.mark.parametrize(
    ("kind", "data_range"),
    [("01", 1.0), ("255", 255.0), ("neg1", 2.0)],
)
def test_meta_records_domain_and_range(
    make_rgb: Callable[..., torch.Tensor],
    kind: str,
    data_range: float,
) -> None:
    torch.manual_seed(2)
    gt = make_rgb(n=2, kind=kind)
    gt_min = float(gt.min().item())
    gt_max = float(gt.max().item())
    pred = (gt + 0.01 * torch.randn_like(gt)).clamp(gt_min, gt_max)

    psnr_meta = rgb_psnr(gt, pred, data_range=data_range, domain="linear", meta=True)
    cpsnr_meta = cpsnr_rgb(gt, pred, data_range=data_range, domain="linear", meta=True)
    ssim_meta = rgb_ssim(gt, pred, data_range=data_range, domain="linear", meta=True)

    assert psnr_meta["meta"]["data_range"] == pytest.approx(data_range)  # type: ignore[index]
    assert cpsnr_meta["meta"]["data_range"] == pytest.approx(data_range)  # type: ignore[index]
    assert ssim_meta["meta"]["data_range"] == pytest.approx(data_range)  # type: ignore[index]
    for meta in (psnr_meta["meta"], cpsnr_meta["meta"], ssim_meta["meta"]):  # type: ignore[index]
        assert meta["domain"] == "linear"  # type: ignore[index]


def test_rgb_psnr_identity_and_monotonic(make_rgb: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(3)
    gt = make_rgb(n=1, kind="01")
    identity = rgb_psnr(gt, gt, data_range=1.0)["mean"]
    assert math.isinf(_as_float(identity)) or _as_float(identity) >= 120.0

    pred_light = (gt + 0.01 * torch.randn_like(gt)).clamp(0.0, 1.0)
    pred_heavy = (gt + 0.05 * torch.randn_like(gt)).clamp(0.0, 1.0)
    psnr_light = rgb_psnr(gt, pred_light, data_range=1.0)["mean"]
    psnr_heavy = rgb_psnr(gt, pred_heavy, data_range=1.0)["mean"]
    assert _as_float(psnr_heavy) <= _as_float(psnr_light) + 1e-5


def test_cpsnr_identity_and_difference(make_rgb: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(4)
    gt = make_rgb(n=1, kind="01")
    noise = torch.zeros_like(gt)
    noise[:, 0].normal_(0.0, 0.06)
    noise[:, 1].normal_(0.0, 0.01)
    noise[:, 2].normal_(0.0, 0.015)
    pred = (gt + noise).clamp(0.0, 1.0)

    psnr_meta = rgb_psnr(gt, pred, data_range=1.0, meta=True)
    cpsnr_meta = cpsnr_rgb(gt, pred, data_range=1.0, meta=True)

    psnr_mean = _as_float(psnr_meta["mean"])
    cpsnr_val = _as_float(cpsnr_meta["cpsnr"])  # type: ignore[index]
    assert abs(psnr_mean - cpsnr_val) > 1e-3

    diff = (gt - pred).to(dtype=torch.float64)
    mse_channels = diff.pow(2).flatten(2).mean(dim=2)
    manual_cpsnr = 10.0 * torch.log10(
        torch.tensor(1.0, dtype=torch.float64) / mse_channels.mean(dim=1).clamp(min=1e-12)
    )
    assert abs(_as_float(manual_cpsnr) - cpsnr_val) <= 1e-6


def test_reduction_modes(make_rgb: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(5)
    gt = make_rgb(n=3)
    pred = (gt + 0.02 * torch.randn_like(gt)).clamp(0.0, 1.0)

    psnr_none = rgb_psnr(gt, pred, data_range=1.0, reduction="none")
    assert psnr_none["R"].shape[0] == gt.shape[0]

    cpsnr_none = cpsnr_rgb(gt, pred, data_range=1.0, reduction="none")
    assert isinstance(cpsnr_none, torch.Tensor)
    assert cpsnr_none.shape[0] == gt.shape[0]

    ssim_none = rgb_ssim(gt, pred, data_range=1.0, reduction="none", channel_aggregate="none")
    assert ssim_none["R"].shape[0] == gt.shape[0]


def test_rgb_ssim_identity_and_mean(make_rgb: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(6)
    gt = make_rgb(n=2, kind="01")
    ssim_mean = rgb_ssim(
        gt,
        gt,
        data_range=1.0,
        kernel_size=11,
        sigma=1.5,
        channel_aggregate="mean",
    )
    assert 0.9999 <= _as_float(ssim_mean["mean"]) <= 1.0001


def test_rgb_ssim_channel_aggregate(make_rgb: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(7)
    gt = make_rgb(n=1)
    pred = (gt + 0.015 * torch.randn_like(gt)).clamp(0.0, 1.0)

    out_none = rgb_ssim(gt, pred, data_range=1.0, channel_aggregate="none")
    out_mean = rgb_ssim(gt, pred, data_range=1.0, channel_aggregate="mean")

    per_channel = [_as_float(out_none[c]) for c in ("R", "G", "B")]
    mean_from_channels = sum(per_channel) / 3.0
    assert abs(mean_from_channels - _as_float(out_none["mean"])) <= 1e-6
    assert abs(mean_from_channels - _as_float(out_mean["mean"])) <= 1e-6


@pytest.mark.parametrize("kernel", [3, 7, 11])
def test_rgb_ssim_kernel_size_and_padding(
    make_rgb: Callable[..., torch.Tensor],
    kernel: int,
) -> None:
    torch.manual_seed(8)
    gt = make_rgb(n=1, h=40, w=40)
    pred = (gt + 0.02 * torch.randn_like(gt)).clamp(0.0, 1.0)
    padding_modes: list[Literal["reflect", "replicate", "circular", "constant"]] = ["reflect", "replicate", "circular", "constant"]
    for padding in padding_modes:
        rgb_ssim(
            gt,
            pred,
            data_range=1.0,
            kernel_size=kernel,
            padding=padding,
            channel_aggregate="none",
        )


def test_rgb_ssim_invalid_kernel(make_rgb: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(9)
    gt = make_rgb(n=1, h=16, w=16)
    pred = gt.clone()
    with pytest.raises(ValueError):
        rgb_ssim(gt, pred, data_range=1.0, kernel_size=10)
    with pytest.raises(ValueError):
        rgb_ssim(gt[:, :, :5, :5], pred[:, :, :5, :5], data_range=1.0, kernel_size=11)


def test_metrics_raise_on_nan(device: torch.device) -> None:
    torch.manual_seed(10)
    gt = torch.rand(1, 3, 16, 16, device=device)
    pred = gt.clone()
    pred[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        rgb_psnr(gt, pred)
    with pytest.raises(ValueError):
        cpsnr_rgb(gt, pred)
    with pytest.raises(ValueError):
        rgb_ssim(gt, pred)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_parity(make_rgb: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(11)
    gt_cpu = make_rgb(n=1, h=24, w=24).cpu()
    pred_cpu = (gt_cpu + 0.02 * torch.randn_like(gt_cpu)).clamp(0.0, 1.0)

    cpu_psnr = rgb_psnr(gt_cpu, pred_cpu, data_range=1.0)["mean"]
    gpu_psnr = rgb_psnr(gt_cpu.cuda(), pred_cpu.cuda(), data_range=1.0)["mean"]
    assert abs(_as_float(cpu_psnr) - _as_float(gpu_psnr)) <= 1e-6

    cpu_cpsnr = cpsnr_rgb(gt_cpu, pred_cpu, data_range=1.0)
    gpu_cpsnr = cpsnr_rgb(gt_cpu.cuda(), pred_cpu.cuda(), data_range=1.0)
    if isinstance(cpu_cpsnr, torch.Tensor) and isinstance(gpu_cpsnr, torch.Tensor):
        assert abs(_as_float(cpu_cpsnr) - _as_float(gpu_cpsnr)) <= 1e-6
    else:
        # CPSNRResult case - extract the scalar value
        assert abs(float(cpu_cpsnr) - float(gpu_cpsnr)) <= 1e-6  # type: ignore[arg-type]

    cpu_ssim = rgb_ssim(gt_cpu, pred_cpu, data_range=1.0, channel_aggregate="mean")["mean"]
    gpu_ssim = rgb_ssim(gt_cpu.cuda(), pred_cpu.cuda(), data_range=1.0, channel_aggregate="mean")["mean"]
    assert abs(_as_float(cpu_ssim) - _as_float(gpu_ssim)) <= 1e-6


def test_rgb_ssim_meta_contains_domain(make_rgb: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(12)
    gt = make_rgb(n=1)
    pred = (gt + 0.01 * torch.randn_like(gt)).clamp(0.0, 1.0)
    result = rgb_ssim(gt, pred, data_range=1.0, meta=True)
    assert result["meta"]["domain"] in {"linear", "srgb"}  # type: ignore[index]
    assert "data_range" in result["meta"]  # type: ignore[operator]

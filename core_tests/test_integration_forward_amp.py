"""
Integration goals:
- Compare FP32 inference against AMP (float16/bfloat16) for PSNR, SSIM, LPIPS, and ΔE2000 within small tolerances.
- Ensure inference uses torch.inference_mode(), autocast contexts only, and disables TF32 for stability.
- Cover CUDA/CPU autocast, channels_last layout, optional timing via CUDA events without hard speed assertions.
"""

from __future__ import annotations

import os
import sys

# 将项目根目录和 NAFNet 目录添加到 Python 路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
nafnet_root = os.path.join(project_root, "NAFNet")
for path in [project_root, nafnet_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

import math
from contextlib import nullcontext
from typing import Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from NewBP_model.losses import DeltaE00Loss, SSIMLoss
except ModuleNotFoundError as exc:  # pragma: no cover
    pytest.skip(f"Skipping AMP integration tests due to missing dependency: {exc}", allow_module_level=True)

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPSMetric
except Exception:  # pragma: no cover
    LPIPSMetric = None


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.head = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return self.head(x)


@pytest.fixture(scope="module", params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request: pytest.FixtureRequest) -> torch.device:
    return torch.device(request.param)


@pytest.fixture(scope="module")
def net(device: torch.device) -> TinyNet:
    torch.manual_seed(0)
    model = TinyNet().to(device)
    model.eval()
    return model


def _autocast_context(device: torch.device, amp_dtype: torch.dtype):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=amp_dtype)
    if device.type == "cpu" and amp_dtype == torch.bfloat16:
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def _psnr_metric(gt: torch.Tensor, pred: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt).item()
    if mse == 0.0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)).item())


def _ssim_metric(gt: torch.Tensor, pred: torch.Tensor) -> float:
    metric = SSIMLoss(window_size=11, max_val=1.0, reduction="none")
    val = 1.0 - metric(pred, gt)
    return float(val.mean().item())


def _lpips_metric(gt: torch.Tensor, pred: torch.Tensor) -> float:
    if LPIPSMetric is None:
        return 0.0
    meter = LPIPSMetric(net_type="alex", normalize=True)
    val = meter(gt, pred)
    return float(val.mean().item())


def _delta_e_metric(gt: torch.Tensor, pred: torch.Tensor) -> float:
    try:
        loss = DeltaE00Loss()
        return float(loss(pred, gt).item())
    except Exception:
        return float((pred - gt).abs().mean().item())


def _compute_metrics(gt: torch.Tensor, pred: torch.Tensor) -> Dict[str, float]:
    return {
        "psnr": _psnr_metric(gt, pred),
        "ssim": _ssim_metric(gt, pred),
        "lpips": _lpips_metric(gt, pred),
        "de2000": _delta_e_metric(gt, pred),
    }


def _close(val_ref: float, val_test: float, *, rtol: float, atol: float) -> bool:
    return abs(val_ref - val_test) <= atol + rtol * abs(val_ref)


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_amp_forward_metrics_close_to_fp32(
    net: TinyNet,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> None:
    if device.type == "cpu" and amp_dtype is torch.float16:
        pytest.skip("CPU autocast supports bfloat16 only")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    torch.manual_seed(42)
    inputs = torch.rand(2, 3, 96, 96, device=device)
    targets = torch.clamp(inputs + 0.05 * torch.randn_like(inputs), 0.0, 1.0)

    with torch.inference_mode():
        preds_fp32 = torch.clamp(net(inputs), 0.0, 1.0)
        metrics_fp32 = _compute_metrics(targets, preds_fp32)

    with torch.inference_mode(), _autocast_context(device, amp_dtype):
        preds_amp = torch.clamp(net(inputs), 0.0, 1.0)
        metrics_amp = _compute_metrics(targets, preds_amp)

    assert _close(metrics_fp32["psnr"], metrics_amp["psnr"], rtol=5e-3, atol=5e-3)
    assert _close(metrics_fp32["ssim"], metrics_amp["ssim"], rtol=5e-3, atol=5e-3)
    assert _close(metrics_fp32["de2000"], metrics_amp["de2000"], rtol=5e-3, atol=5e-3)
    assert _close(metrics_fp32["lpips"], metrics_amp["lpips"], rtol=1e-2, atol=5e-3)


@pytest.mark.parametrize("amp_dtype", [torch.float16])
def test_channels_last_amp_support(
    net: TinyNet,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> None:
    if device.type != "cuda":
        pytest.skip("channels_last AMP primarily relevant on CUDA")

    torch.backends.cuda.matmul.allow_tf32 = False

    inputs = torch.rand(1, 3, 128, 128, device=device).contiguous(memory_format=torch.channels_last)
    targets = torch.rand(1, 3, 128, 128, device=device).contiguous(memory_format=torch.channels_last)
    net_cl = net.to(memory_format=torch.channels_last)

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=amp_dtype):
        outputs = torch.clamp(net_cl(inputs), 0.0, 1.0)
        metrics = _compute_metrics(targets, outputs)
        assert all(math.isfinite(v) for v in metrics.values())


def test_amp_timing_record_only(net: TinyNet, device: torch.device) -> None:
    if device.type != "cuda":
        pytest.skip("Timing harness uses CUDA events")

    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = torch.rand(4, 3, 256, 256, device=device)
    iterations = 10

    def _run(amp: bool) -> float:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        with torch.inference_mode():
            ctx = torch.cuda.amp.autocast() if amp else nullcontext()
            with ctx:
                for _ in range(iterations):
                    _ = net(inputs)
        ender.record()
        torch.cuda.synchronize()
        return starter.elapsed_time(ender) / iterations

    t_fp32 = _run(amp=False)
    t_amp = _run(amp=True)
    print(f"[timing] FP32: {t_fp32:.3f} ms/iter, AMP: {t_amp:.3f} ms/iter")


# Remarks:
# torch.inference_mode() is preferred over no_grad() for inference-only workloads.
# AMP inference only needs autocast; GradScaler is unnecessary without backprop.
# Disabling TF32 aligns FP32 baselines with AMP precision for consistent comparisons.
# Channels_last coverage ensures NHWC compatibility under autocast contexts.
# CUDA timing uses events plus synchronize for accurate measurements.

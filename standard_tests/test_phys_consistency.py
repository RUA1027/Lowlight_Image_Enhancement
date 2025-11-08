"""
Physics-consistency regression tests for RAW and sRGB metrics.

Verified properties reference:
- Convolutional imaging with PSF under shift-invariance (point spread function,
  see standard optics references).
- Exposure linearity in RAW domain allowing scaling by the exposure ratio.
- sRGB transfer characteristics with clamp-to-[0,1] when comparing in display space.
- Charbonnier (pseudo-Huber) robust penalty smoothly bridging L1/L2 behaviours.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../NAFNet")))

import math
from typing import Callable, Tuple

import pytest
import torch

from metrics.phys_consistency import phys_cons_raw, phys_cons_srgb


def _make_delta_psf(channels: int, kernel: int = 3, device: torch.device | None = None) -> torch.Tensor:
    psf = torch.zeros(channels, channels, kernel, kernel, device=device)
    centre = kernel // 2
    eye = torch.eye(channels, device=device)
    for o in range(channels):
        for i in range(channels):
            psf[o, i, centre, centre] = eye[o, i]
    return psf


def _make_blur_psf(channels: int, kernel: int, device: torch.device) -> torch.Tensor:
    psf = torch.ones(channels, channels, kernel, kernel, device=device)
    psf = psf / psf.view(channels, -1).sum(dim=1).view(channels, 1, 1, 1)
    return psf


@pytest.fixture(scope="module", params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request: pytest.FixtureRequest) -> torch.device:
    return torch.device(request.param)


@pytest.fixture()
def make_tensor(device: torch.device) -> Callable[[int, int, int, int, str], torch.Tensor]:
    def _factory(n: int, c: int, h: int, w: int, domain: str = "raw") -> torch.Tensor:
        data = torch.rand(n, c, h, w, device=device)
        if domain == "srgb":
            data = data.clamp(0.0, 1.0)
        return data.to(dtype=torch.float32)

    return _factory


def _metric_value(output: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]) -> float:
    if isinstance(output, tuple):
        metric, _ = output
    else:
        metric = output
    return float(metric.detach().cpu().item())


def _map_shape(output: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[int, ...]:
    return tuple(output[1].shape)


def test_psf_shape_validation(device: torch.device, make_tensor: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(0)
    data = make_tensor(1, 3, 24, 24, "raw")
    psf_even = torch.ones(3, 3, 4, 4, device=device)
    with pytest.raises(ValueError):
        phys_cons_raw(data, data, psf_even, expo_ratio=1.0)

    psf_bad_channels = torch.ones(2, 3, 3, 3, device=device)
    with pytest.raises(ValueError):
        phys_cons_raw(data, data, psf_bad_channels, expo_ratio=1.0)


def test_zero_error_delta_psf(device: torch.device, make_tensor: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(1)
    gt = make_tensor(1, 3, 32, 32, "raw")
    rho = 0.7
    pred = gt.clone()
    obs = rho * gt
    psf = _make_delta_psf(3, kernel=3, device=device)

    metric = phys_cons_raw(pred, obs, psf, expo_ratio=rho, crop="same", padding="reflect", normalize_psf=True)
    assert abs(_metric_value(metric)) <= 1e-7


def test_crop_and_padding_semantics(device: torch.device, make_tensor: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(2)
    gt = make_tensor(1, 3, 40, 40, "raw")
    psf = _make_delta_psf(3, kernel=5, device=device)

    same_metric_result = phys_cons_raw(gt, gt, psf, expo_ratio=1.0, crop="same", padding="replicate", return_map=True)
    assert isinstance(same_metric_result, tuple)
    assert _map_shape(same_metric_result) == (1, 3, 40, 40)

    valid_metric_result = phys_cons_raw(gt, gt, psf, expo_ratio=1.0, crop="valid", padding="zeros", return_map=True)
    assert isinstance(valid_metric_result, tuple)
    assert _map_shape(valid_metric_result) == (1, 3, 36, 36)

    for padding in ("reflect", "replicate", "zeros"):
        out = phys_cons_raw(gt, gt, psf, expo_ratio=1.0, crop="same", padding=padding)
        assert math.isfinite(_metric_value(out))


@pytest.mark.parametrize("kernel", [3, 5, 9])
def test_monotonic_blur_and_exposure(
    device: torch.device,
    make_tensor: Callable[..., torch.Tensor],
    kernel: int,
) -> None:
    torch.manual_seed(3)
    pred = make_tensor(1, 3, 48, 48, "raw")
    obs = pred.clone()

    psf_small = _make_blur_psf(3, 3, device)
    psf_large = _make_blur_psf(3, kernel, device)

    small_val = _metric_value(phys_cons_raw(pred, obs, psf_small, expo_ratio=1.0, crop="same"))
    large_val = _metric_value(phys_cons_raw(pred, obs, psf_large, expo_ratio=1.0, crop="same"))
    tol = 1e-3
    assert large_val + tol >= small_val

    rho_low = _metric_value(phys_cons_raw(pred, obs, psf_small, expo_ratio=0.5, crop="same"))
    rho_high = _metric_value(phys_cons_raw(pred, obs, psf_small, expo_ratio=1.5, crop="same"))
    assert rho_high + tol >= rho_low


@pytest.mark.parametrize("eps", [1e-6, 1e-3])
def test_charbonnier_vs_l1(device: torch.device, make_tensor: Callable[..., torch.Tensor], eps: float) -> None:
    torch.manual_seed(4)
    gt = make_tensor(1, 3, 32, 32, "raw")
    pred = (gt + 0.02 * torch.randn_like(gt)).clamp(0.0, 1.0)
    psf = _make_delta_psf(3, device=device)

    l1_val = _metric_value(phys_cons_raw(pred, gt, psf, expo_ratio=1.0, robust="none", crop="same"))
    charb_val = _metric_value(
        phys_cons_raw(pred, gt, psf, expo_ratio=1.0, robust="charbonnier", crop="same", eps=eps)
    )
    assert charb_val >= l1_val


def test_srgb_clamp_behavior(device: torch.device) -> None:
    torch.manual_seed(5)
    pred = torch.full((1, 3, 24, 24), 0.9, device=device)
    obs = torch.full((1, 3, 24, 24), 0.8, device=device)
    psf = _make_delta_psf(3, device=device)
    ratio = 1.5

    unclamped = phys_cons_srgb(pred, obs, psf, expo_ratio=ratio, clamp01=False, crop="same")
    clamped = phys_cons_srgb(pred, obs, psf, expo_ratio=ratio, clamp01=True, crop="same", return_map=True)

    assert _metric_value(clamped) <= _metric_value(unclamped)
    assert torch.max(clamped[1]) <= 0.25 + 1e-6  # clamp to 1.0 limits residuals


@pytest.mark.parametrize("shape", ["scalar", "batch", "spatial", "full"])
def test_exposure_ratio_broadcast(
    device: torch.device,
    make_tensor: Callable[..., torch.Tensor],
    shape: str,
) -> None:
    torch.manual_seed(6)
    gt = make_tensor(2, 3, 28, 28, "raw")
    pred = make_tensor(2, 3, 28, 28, "raw")
    psf = _make_delta_psf(3, device=device)

    if shape == "scalar":
        rho = 0.8
    elif shape == "batch":
        rho = torch.tensor([0.8, 1.2], device=device)
    elif shape == "spatial":
        rho = torch.full((2, 1, 28, 28), 0.9, device=device)
    else:
        rho = torch.full((2, 3, 28, 28), 1.1, device=device)

    output = phys_cons_raw(pred, gt, psf, expo_ratio=rho, crop="same")
    assert math.isfinite(_metric_value(output))


def test_zero_psf_triggers_warning(device: torch.device, make_tensor: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(7)
    gt = make_tensor(1, 3, 20, 20, "raw")
    zero_psf = torch.zeros(3, 3, 3, 3, device=device)
    with pytest.warns(RuntimeWarning):
        val = phys_cons_raw(gt, gt, zero_psf, expo_ratio=1.0, normalize_psf=True)
    assert math.isfinite(_metric_value(val))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_parity():
    torch.manual_seed(8)
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda")
    data_cpu = torch.rand(1, 3, 18, 18, device=device_cpu)
    obs_cpu = torch.rand_like(data_cpu)
    psf_cpu = _make_blur_psf(3, 5, device_cpu)

    cpu_val = _metric_value(phys_cons_raw(data_cpu, obs_cpu, psf_cpu, expo_ratio=1.0, crop="same"))
    gpu_val = _metric_value(
        phys_cons_raw(data_cpu.cuda(), obs_cpu.cuda(), psf_cpu.cuda(), expo_ratio=1.0, crop="same")
    )
    assert abs(cpu_val - gpu_val) <= 1e-6

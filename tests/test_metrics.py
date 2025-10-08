"""Unit tests for metric utilities (PSNR, SSIM, LPIPS, parameter count)."""

import math

import pytest
import torch

from metrics.psnr import PSNRMetric, calculate_psnr, calculate_psnr_per_image
from metrics.ssim import SSIMMetric, calculate_ssim, calculate_ssim_per_image
from metrics.lpips_metric import LPIPSEvaluator
from metrics.parameter_utils import count_parameters
from metrics.flops_utils import count_flops
from metrics.inference_time import measure_inference_time


@torch.no_grad()
def _make_images(batch: int, channels: int, height: int, width: int, noise_scale: float = 0.0):
    gt = torch.rand(batch, channels, height, width)
    pred = gt.clone()
    if noise_scale > 0.0:
        pred = torch.clamp(pred + noise_scale * torch.randn_like(pred), 0.0, 1.0)
    return gt, pred


def test_psnr_single_and_batch():
    gt, pred = _make_images(1, 3, 16, 16, noise_scale=0.01)
    val_single = calculate_psnr(gt[0], pred[0], data_range=1.0)
    assert isinstance(val_single, float)
    assert val_single > 20.0

    vals_batch = calculate_psnr_per_image(gt, pred, data_range=1.0)
    assert vals_batch.shape == (1,)
    assert torch.isfinite(vals_batch).all()

    identical = calculate_psnr_per_image(gt, gt, data_range=1.0)
    assert identical[0] == float("inf")

    metric = PSNRMetric(data_range=1.0)
    metric.update(gt, pred)
    stats = metric.compute()
    assert stats["count"] == 1.0
    assert stats["mean"] > 20.0
    assert stats["num_infinite"] == 0.0

    metric.update(gt, gt)
    stats = metric.compute()
    assert stats["count"] == 2.0
    assert stats["num_infinite"] == 1.0


def test_ssim_single_and_batch():
    gt, pred = _make_images(2, 1, 32, 32, noise_scale=0.1)
    val_single = calculate_ssim(gt[0], pred[0], data_range=1.0)
    assert isinstance(val_single, float)
    assert -1.0 <= val_single <= 1.0

    vals_batch = calculate_ssim_per_image(gt, pred, data_range=1.0)
    assert vals_batch.shape == (2,)
    assert torch.all(vals_batch <= 1.0)

    metric = SSIMMetric(data_range=1.0)
    metric.update(gt, pred)
    stats = metric.compute()
    assert stats["count"] == 2.0
    assert -1.0 <= stats["mean"] <= 1.0


def test_lpips_evaluator_single_and_aggregate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = LPIPSEvaluator(net="alex", device=device)
    evaluator.reset()

    gt, pred = _make_images(3, 3, 32, 32, noise_scale=0.05)
    score_single = evaluator(gt, pred)
    assert isinstance(score_single, float)
    assert score_single >= 0.0

    evaluator.update(gt, pred)
    evaluator.update(gt, gt)
    stats = evaluator.compute()
    assert stats["count"] == 6.0
    assert stats["mean"] >= 0.0
    assert stats["std"] >= 0.0


def test_count_parameters_units():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(10, 5))
            self.bias = torch.nn.Parameter(torch.zeros(10))
            self.frozen = torch.nn.Parameter(torch.ones(3), requires_grad=False)

        def forward(self, x):  # pragma: no cover - not used
            return x

    model = TinyModel()
    raw = count_parameters(model, unit="raw")
    assert raw == 10 * 5 + 10
    assert math.isclose(count_parameters(model, unit="K"), raw / 1e3)
    assert math.isclose(count_parameters(model, unit="M"), raw / 1e6)


def test_count_flops_requires_fvcore():
    pytest.importorskip("fvcore")

    class SmallNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, kernel_size=3)

        def forward(self, x):  # pragma: no cover - simple forward
            return self.conv(x)

    model = SmallNet()
    input_tensor = torch.randn(1, 3, 32, 32)
    flops_g = count_flops(model, input_tensor, unit="G")
    assert flops_g > 0
    with pytest.raises(ValueError):
        count_flops(model, input_tensor, unit="T")


def test_measure_inference_time_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for inference time benchmark.")

    device = torch.device("cuda")

    class TinyConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(8, 8, kernel_size=3, padding=1),
            )

        def forward(self, x):  # pragma: no cover - simple forward
            return self.net(x)

    model = TinyConv().to(device)
    dummy_input = torch.randn(2, 3, 64, 64, device=device)
    avg_time = measure_inference_time(model, dummy_input, num_warmup=2, num_runs=5)
    assert avg_time > 0

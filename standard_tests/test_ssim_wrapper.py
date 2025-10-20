"""
SSIM wrapper regression tests enforcing literature-aligned configuration.

Key references:
- Wang et al. (2004): single-scale SSIM with Gaussian window (kernel_size=11, sigma=1.5, k1=0.01, k2=0.03).
- TorchMetrics StructuralSimilarityIndexMeasure: authoritative parameter names/defaults (`kernel_size`, `sigma`, etc.).
- scikit-image metrics documentation: float images must supply correct `data_range`, window size must be odd and not exceed the image extent.
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

import pytest
import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

from metrics.ssim import SSIMEvaluator, SSIMMetric, evaluate_pairs_ssim


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
            x = (x * 255.0).round()
        elif kind == "neg1":
            x = x * 2.0 - 1.0
        return x.to(dtype=torch.float32)

    return _factory


def _mean_from_summary(summary: dict) -> float:
    return float(summary["mean"])


@pytest.mark.parametrize(
    ("kind", "expected_range"),
    [("01", 1.0), ("255", 255.0), ("neg1", 2.0)],
)
def test_auto_vs_explicit_data_range(
    device: torch.device,
    make_images: Callable[..., torch.Tensor],
    kind: str,
    expected_range: float,
) -> None:
    torch.manual_seed(0)
    gt = make_images(n=2, c=3, h=40, w=40, kind=kind)
    pred = (gt + 0.02 * torch.randn_like(gt)).clone()
    if kind == "01":
        pred = pred.clamp(0.0, 1.0)
    elif kind == "255":
        pred = pred.clamp(0.0, 255.0)
    else:
        pred = pred.clamp(-1.0, 1.0)

    evaluator_auto = SSIMEvaluator(kernel_size=11, sigma=1.5, resize_policy=None)
    stats_auto = evaluator_auto(gt, pred)

    evaluator_fixed = SSIMEvaluator(kernel_size=11, sigma=1.5, data_range=expected_range)
    stats_fixed = evaluator_fixed(gt, pred)

    assert abs(_mean_from_summary(stats_auto) - _mean_from_summary(stats_fixed)) <= 1e-6
    assert stats_auto["data_range"] == pytest.approx(expected_range)


def test_size_mismatch_and_resize_policies(make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(1)
    gt = make_images(n=1, c=3, h=80, w=80)
    pred = F.interpolate(gt, size=(48, 60), mode="bilinear", align_corners=False)

    evaluator = SSIMEvaluator(kernel_size=11, sigma=1.5, resize_policy=None)
    with pytest.raises(ValueError):
        evaluator(gt, pred)

    eval_resize = SSIMEvaluator(kernel_size=11, sigma=1.5, resize_policy="resize")
    stats_resize = eval_resize(gt, pred)
    assert stats_resize["resize_policy"] == "resize"
    assert math.isfinite(stats_resize["mean"])

    eval_crop = SSIMEvaluator(kernel_size=11, sigma=1.5, resize_policy="center_crop")
    stats_crop = eval_crop(gt, pred)
    assert stats_crop["resize_policy"] == "center_crop"
    assert math.isfinite(stats_crop["mean"])


def test_identity_and_noise_monotonicity(make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(2)
    gt = make_images(n=1, c=3, h=64, w=64)
    evaluator = SSIMEvaluator(kernel_size=11, sigma=1.5, data_range=1.0)

    same = _mean_from_summary(evaluator(gt, gt))
    assert 0.9999 <= same <= 1.0001

    pred_small = (gt + 0.01 * torch.randn_like(gt)).clamp(0.0, 1.0)
    pred_large = (gt + 0.05 * torch.randn_like(gt)).clamp(0.0, 1.0)
    small_score = _mean_from_summary(evaluator(gt, pred_small))
    large_score = _mean_from_summary(evaluator(gt, pred_large))
    assert large_score <= small_score + 1e-6


@pytest.mark.parametrize("kernel_size", [3, 7, 11, 21])
def test_kernel_size_adaptation(
    make_images: Callable[..., torch.Tensor],
    kernel_size: int,
) -> None:
    torch.manual_seed(3)
    gt = make_images(n=1, c=3, h=16, w=20)
    pred = (gt + 0.01 * torch.randn_like(gt)).clamp(0.0, 1.0)
    evaluator = SSIMEvaluator(kernel_size=kernel_size, sigma=1.5, data_range=1.0)
    stats = evaluator(gt, pred)
    assert stats["kernel_size"] % 2 == 1
    assert stats["kernel_size"] <= min(gt.shape[-2:])


def test_matches_torchmetrics_reference(device: torch.device, make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(4)
    gt = make_images(n=2, c=3, h=48, w=48, kind="01")
    pred = (gt + 0.02 * torch.randn_like(gt)).clamp(0.0, 1.0)

    evaluator = SSIMEvaluator(kernel_size=11, sigma=1.5, data_range=1.0, resize_policy=None)
    ours = _mean_from_summary(evaluator(gt, pred))

    reference = StructuralSimilarityIndexMeasure(
        gaussian_kernel=True,
        sigma=1.5,
        kernel_size=11,
        data_range=1.0,
        k1=0.01,
        k2=0.03,
    ).to(device).eval()
    with torch.no_grad():
        ref_val = float(reference(pred.to(device), gt.to(device)).item())
    assert abs(ours - ref_val) <= 1e-6


def test_statistics_and_quantiles(make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(5)
    gt = make_images(n=5, c=3, h=36, w=36)
    pred = (gt + 0.03 * torch.randn_like(gt)).clamp(0.0, 1.0)

    evaluator = SSIMEvaluator(kernel_size=11, sigma=1.5, data_range=1.0)
    stats = evaluator(gt, pred)

    per_image = torch.tensor(stats["per_image"], dtype=torch.float32)
    assert abs(per_image.mean().item() - stats["mean"]) <= 1e-7
    if hasattr(torch, "quantile"):
        assert abs(float(torch.quantile(per_image, 0.5).item()) - stats["p50"]) <= 1e-6
        assert abs(float(torch.quantile(per_image, 0.95).item()) - stats["p95"]) <= 1e-6


def test_evaluate_pairs_consistency(make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(6)
    imgs = make_images(n=4, c=3, h=32, w=32)
    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    evaluator = SSIMEvaluator(kernel_size=11, sigma=1.5, data_range=1.0)
    manual_scores: List[float] = []
    for idx in range(imgs.shape[0]):
        gt = imgs[idx : idx + 1]
        pred = (gt + 0.02 * torch.randn_like(gt)).clamp(0.0, 1.0)
        pairs.append((gt.squeeze(0), pred.squeeze(0)))
        manual_scores.extend(evaluator(gt, pred)["per_image"])

    summary = evaluate_pairs_ssim(pairs, kernel_size=11, sigma=1.5, data_range=1.0)
    assert abs(sum(manual_scores) / len(manual_scores) - summary["mean"]) <= 1e-6
    assert summary["count"] == len(manual_scores)


def test_color_space_modes(make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(7)
    gt = make_images(n=1, c=3, h=40, w=40)
    pred = (gt + 0.02 * torch.randn_like(gt)).clamp(0.0, 1.0)

    eval_rgb = SSIMEvaluator(kernel_size=11, sigma=1.5, data_range=1.0, color_space="rgb")
    eval_y = SSIMEvaluator(kernel_size=11, sigma=1.5, data_range=1.0, color_space="y")

    rgb_stats = eval_rgb(gt, pred)
    y_stats = eval_y(gt, pred)
    assert math.isfinite(rgb_stats["mean"])
    assert math.isfinite(y_stats["mean"])

    identity_y = _mean_from_summary(eval_y(gt, gt))
    assert identity_y >= 0.9999


def test_ssimmetric_streaming(make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(8)
    gt = make_images(n=3, c=3, h=32, w=32, kind="01")
    preds = (gt + 0.02 * torch.randn_like(gt)).clamp(0.0, 1.0)

    metric = SSIMMetric(data_range=1.0, kernel_size=11, sigma=1.5)
    evaluator = SSIMEvaluator(kernel_size=11, sigma=1.5, data_range=1.0)

    all_scores = []
    for idx in range(gt.shape[0]):
        g = gt[idx : idx + 1]
        p = preds[idx : idx + 1]
        metric.update(g, p)
        all_scores.extend(evaluator(g, p)["per_image"])

    summary = metric.compute()
    assert abs(summary["mean"] - sum(all_scores) / len(all_scores)) <= 1e-6
    assert summary["count"] == len(all_scores)


def test_small_images_and_kernel_adjustment(make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(9)
    gt = make_images(n=1, c=3, h=11, w=11)
    pred = (gt + 0.01 * torch.randn_like(gt)).clamp(0.0, 1.0)
    evaluator = SSIMEvaluator(kernel_size=21, sigma=1.5, data_range=1.0)
    stats = evaluator(gt, pred)
    assert stats["kernel_size"] == 11
    assert math.isfinite(stats["mean"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_parity(make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(10)
    gt_cpu = make_images(n=1, c=3, h=28, w=28).cpu()
    pred_cpu = (gt_cpu + 0.02 * torch.randn_like(gt_cpu)).clamp(0.0, 1.0)

    evaluator = SSIMEvaluator(kernel_size=11, sigma=1.5, data_range=1.0)
    cpu_stats = evaluator(gt_cpu, pred_cpu)
    gpu_stats = evaluator(gt_cpu.cuda(), pred_cpu.cuda())
    assert abs(cpu_stats["mean"] - gpu_stats["mean"]) <= 1e-6


def test_resize_policy_metadata(make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(11)
    gt = make_images(n=1, c=3, h=64, w=64)
    pred = F.interpolate(gt, size=(52, 60), mode="bilinear", align_corners=False)

    evaluator = SSIMEvaluator(kernel_size=11, sigma=1.5, data_range=1.0, resize_policy="resize")
    stats = evaluator(gt, pred)
    assert stats["resize_policy"] == "resize"


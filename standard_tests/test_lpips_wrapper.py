"""
LPIPS wrapper regression tests aligned with official implementation.

References:
- richzhang/PerceptualSimilarity (official LPIPS repo): expects RGB inputs in
  shape [N,3,H,W], normalised to [-1,1]; backbones include alex/vgg/squeeze and
  version '0.1'; lower scores indicate higher perceptual similarity.
- TorchMetrics LPIPS: documents the [-1,1] convention, optional normalisation
  from [0,1], and minimum spatial requirements depending on the backbone.
- Zhang et al. (CVPR 2018): theoretical foundation describing LPIPS behaviour.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../NAFNet")))

import importlib.util
import math
import pathlib
from typing import Callable, List, Tuple

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("lpips")
import lpips  # noqa: E402  (import after importorskip)


ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lpips_metric_module", ROOT / "metrics" / "lpips_metric.py")
lpips_module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
if spec is not None and spec.loader is not None:  # type: ignore[union-attr]
    spec.loader.exec_module(lpips_module)  # type: ignore[arg-type]

LPIPSMetric = lpips_module.LPIPSMetric
LPIPSEvaluator = lpips_module.LPIPSEvaluator
evaluate_pairs = lpips_module.evaluate_pairs


@pytest.fixture(scope="module", params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request: pytest.FixtureRequest) -> torch.device:
    return torch.device(request.param)


@pytest.fixture()
def make_images(device: torch.device) -> Callable[..., torch.Tensor]:
    def _factory(
        n: int = 2,
        c: int = 3,
        h: int = 64,
        w: int = 64,
        kind: str = "01",
    ) -> torch.Tensor:
        x = torch.rand(n, c, h, w, device=device)
        if kind == "255":
            x = (x * 255.0).round()
        elif kind == "neg1":
            x = x * 2.0 - 1.0
        return x.to(dtype=torch.float32)

    return _factory


@pytest.mark.parametrize("kind", ["01", "255", "neg1"])
def test_value_ranges_and_gray_support(
    device: torch.device,
    make_images: Callable[..., torch.Tensor],
    kind: str,
) -> None:
    torch.manual_seed(0)
    gt = make_images(n=2, c=3, h=48, w=48, kind=kind)
    gray = make_images(n=2, c=1, h=48, w=48, kind=kind)

    metric = LPIPSMetric(net="alex", version="0.1", resize_policy=None, reduce="none")
    same = metric(gt, gt)
    assert same["mean"] <= 1e-6

    # 灰度与RGB混用应抛出异常
    import pytest
    with pytest.raises(ValueError):
        metric(gt, gray)
    
    # 测试灰度图像自身的LPIPS计算
    gray_stats = metric(gray, gray)
    assert gray_stats["mean"] <= 1e-6
    
    assert "net" in same and same["net"] == "alex"
    assert "version" in same and same["version"] == "0.1"
    assert isinstance(gray_stats["per_image"], list) and len(gray_stats["per_image"]) == gray.shape[0]


def test_size_alignment_policies(make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(1)
    gt = make_images(n=1, c=3, h=96, w=96, kind="01")
    pred = F.interpolate(gt, size=(64, 80), mode="bilinear", align_corners=False)

    metric = LPIPSMetric(net="alex", version="0.1", resize_policy=None)
    with pytest.raises(ValueError):
        metric(gt, pred)

    resize_metric = LPIPSMetric(net="alex", version="0.1", resize_policy="resize")
    stats_resize = resize_metric(gt, pred)
    assert stats_resize["resize_policy"] == "resize"
    assert math.isfinite(stats_resize["mean"])

    crop_metric = LPIPSMetric(net="alex", version="0.1", resize_policy="center_crop")
    stats_crop = crop_metric(gt, pred)
    assert stats_crop["resize_policy"] == "center_crop"
    assert math.isfinite(stats_crop["mean"])


def test_spatial_map_consistency(make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(2)
    gt = make_images(n=1, c=3, h=72, w=72, kind="01")
    pred = (gt + 0.02 * torch.randn_like(gt)).clamp(0.0, 1.0)

    metric = LPIPSMetric(net="alex", version="0.1", resize_policy=None)
    scalar_stats = metric(gt, pred)

    t, p = metric._prepare_images(gt, pred)  # type: ignore[attr-defined]
    spatial_model = lpips.LPIPS(net="alex", version="0.1", spatial=True).to(metric.device)
    spatial_model.eval()
    with torch.no_grad():
        spatial_map = spatial_model(p, t)
    spatial_map = spatial_map.mean(dim=1)  # [N,H,W]
    spatial_mean = float(spatial_map.mean().item())
    assert abs(spatial_mean - scalar_stats["mean"]) <= 1e-3


def test_device_fallback_and_cache(device: torch.device, make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(3)
    gt = make_images(n=1, c=3, h=40, w=40, kind="01")
    evaluator = LPIPSEvaluator(net="alex")
    score_eval = evaluator(gt, gt)
    assert math.isclose(score_eval, 0.0, abs_tol=1e-6)

    metric = LPIPSMetric(net="alex", version="0.1")
    assert metric._lpips is None
    first = metric(gt, gt)
    cached = metric._lpips
    second = metric(gt, gt)
    assert cached is metric._lpips
    assert math.isclose(first["mean"], second["mean"], abs_tol=1e-9)


def test_batch_statistics(make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(4)
    gt = make_images(n=5, c=3, h=64, w=64, kind="01")
    pred = (gt + 0.03 * torch.randn_like(gt)).clamp(0.0, 1.0)
    metric = LPIPSMetric(net="alex", version="0.1", reduce="none")
    stats = metric(gt, pred)

    per = torch.tensor(stats["per_image"], dtype=torch.float32)
    assert abs(per.mean().item() - stats["mean"]) <= 1e-7
    if hasattr(torch, "quantile"):
        assert abs(float(torch.quantile(per, 0.5).item()) - stats["p50"]) <= 1e-6
        assert abs(float(torch.quantile(per, 0.95).item()) - stats["p95"]) <= 5e-4


def test_monotonicity_and_backbones(make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(5)
    gt = make_images(n=2, c=3, h=96, w=96, kind="01")
    pred_a = (gt + 0.01 * torch.randn_like(gt)).clamp(0.0, 1.0)
    pred_b = (gt + 0.05 * torch.randn_like(gt)).clamp(0.0, 1.0)

    alex_metric = LPIPSMetric(net="alex", version="0.1")
    vgg_metric = LPIPSMetric(net="vgg", version="0.1")

    s_a_alex = alex_metric(gt, pred_a)["mean"]
    s_b_alex = alex_metric(gt, pred_b)["mean"]
    s_a_vgg = vgg_metric(gt, pred_a)["mean"]
    s_b_vgg = vgg_metric(gt, pred_b)["mean"]

    assert s_b_alex >= s_a_alex - 1e-4
    assert s_b_vgg >= s_a_vgg - 1e-4


def test_evaluate_pairs_vs_manual(make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(6)
    imgs = make_images(n=4, c=3, h=64, w=64, kind="01")
    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    metric = LPIPSMetric(net="alex", version="0.1", reduce="none")
    manual: List[float] = []
    for idx in range(imgs.shape[0]):
        gt = imgs[idx : idx + 1]
        pred = (gt + 0.02 * torch.randn_like(gt)).clamp(0.0, 1.0)
        pairs.append((gt.squeeze(0), pred.squeeze(0)))
        manual.extend(metric(gt, pred)["per_image"])

    summary = evaluate_pairs(pairs, net="alex", version="0.1")
    assert abs(sum(manual) / len(manual) - summary["mean"]) <= 1e-6
    assert summary["count"] == len(manual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_amp_autocast_stability(make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(7)
    gt = make_images(n=1, c=3, h=72, w=72, kind="01").cuda()
    pred = (gt + 0.02 * torch.randn_like(gt)).clamp(0.0, 1.0)
    metric = LPIPSMetric(net="alex", version="0.1", use_amp=True)

    with torch.cuda.amp.autocast(dtype=torch.float16):  # type: ignore[attr-defined]
        amp_val = metric(gt, pred)["mean"]
    fp32_val = metric(gt.float(), pred.float())["mean"]
    assert math.isfinite(amp_val)
    assert abs(amp_val - fp32_val) <= 1e-3


def test_lpips_evaluator_gray_channel(make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(8)
    gt = make_images(n=1, c=1, h=48, w=48, kind="01")
    evaluator = LPIPSEvaluator(net="alex")
    val = evaluator(gt, gt)
    assert math.isclose(val, 0.0, abs_tol=1e-6)


@pytest.mark.parametrize("shape", ["scalar", "batch", "image"])
def test_resize_policy_metadata(make_images: Callable[..., torch.Tensor], shape: str) -> None:
    torch.manual_seed(9)
    gt = make_images(n=2, c=3, h=64, w=64, kind="01")
    pred = F.interpolate(gt, size=(48, 48), mode="bilinear", align_corners=False)
    resize_metric = LPIPSMetric(net="alex", version="0.1", resize_policy="resize")
    stats = resize_metric(gt, pred)
    assert stats["resize_policy"] == "resize"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_parity(make_images: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(10)
    gt_cpu = make_images(n=1, c=3, h=56, w=56).cpu()
    pred_cpu = (gt_cpu + 0.02 * torch.randn_like(gt_cpu)).clamp(0.0, 1.0)

    metric = LPIPSMetric(net="alex", version="0.1")
    cpu_val = metric(gt_cpu, pred_cpu)["mean"]
    gpu_val = metric(gt_cpu.cuda(), pred_cpu.cuda())["mean"]
    assert abs(cpu_val - gpu_val) <= 1e-6

"""Wrappers around project-level metrics for integration with BasicSR validation.

This module bridges the BasicSR-internal metrics namespace (``basicsr.metrics``)
with the project-root ``metrics`` package that lives outside ``NAFNet_base``.

Some environments have import resolution quirks when a package named
``basicsr.metrics`` is importing a top-level package also named ``metrics``.
To make this robust, we fall back to loading the project's metrics modules by
absolute file path if the regular import fails.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import sys
from pathlib import Path
import torch
import importlib.util

# Try the straightforward absolute import first. If that fails, load by file path.
_current = Path(__file__).resolve()
_repo_root: Optional[Path] = None
try:
    # .../NAFNet_base/basicsr/metrics/lowlight_metrics.py -> repo root at parents[3]
    candidate = _current.parents[3]
    if (candidate / 'metrics' / 'color_error.py').exists():
        _repo_root = candidate
except Exception:
    _repo_root = None

def _load_by_path(module_name: str, file_path: Path):
    """Load a module from an explicit file path with a unique name."""
    unique_name = f"project_metrics__{module_name}"
    if unique_name in sys.modules:
        return sys.modules[unique_name]
    spec = importlib.util.spec_from_file_location(unique_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

try:
    # Preferred: normal import assuming project root is on sys.path
    from metrics.color_error import deltaE2000_summary, edge_deltaE2000  # type: ignore
    from metrics.lpips_metric import LPIPSEvaluator  # type: ignore
    from metrics.psnr import calculate_psnr  # type: ignore
    from metrics.ssim import calculate_ssim  # type: ignore
except ModuleNotFoundError:
    if _repo_root is None:
        raise
    # Fallback: load from absolute file paths to avoid package name collisions
    _color_error = _load_by_path('color_error', _repo_root / 'metrics' / 'color_error.py')
    _lpips_metric = _load_by_path('lpips_metric', _repo_root / 'metrics' / 'lpips_metric.py')
    _psnr = _load_by_path('psnr', _repo_root / 'metrics' / 'psnr.py')
    _ssim = _load_by_path('ssim', _repo_root / 'metrics' / 'ssim.py')

    deltaE2000_summary = getattr(_color_error, 'deltaE2000_summary')
    edge_deltaE2000 = getattr(_color_error, 'edge_deltaE2000')
    LPIPSEvaluator = getattr(_lpips_metric, 'LPIPSEvaluator')
    calculate_psnr = getattr(_psnr, 'calculate_psnr')
    calculate_ssim = getattr(_ssim, 'calculate_ssim')


def linear_psnr(pred: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0) -> float:
    pred = pred.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)
    return calculate_psnr(target, pred, data_range=data_range)


def linear_ssim(pred: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0) -> float:
    pred = pred.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)
    return calculate_ssim(target, pred, data_range=data_range)


def lpips_distance(pred: torch.Tensor, target: torch.Tensor, *, net: str = "vgg", device: Optional[str] = None) -> float:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = LPIPSEvaluator(net=net, device=dev)
    return evaluator(target.to(dev), pred.to(dev))


def deltae2000_mean(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    whitepoint: Literal["D65-2", "D50-2"] = "D65-2",
) -> float:
    summary = deltaE2000_summary(
        pred.clamp(0.0, 1.0),
        target.clamp(0.0, 1.0),
        whitepoint=whitepoint,
        percentiles=(95.0,),
    )
    return summary["mean"]


def deltae2000_p95(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    whitepoint: Literal["D65-2", "D50-2"] = "D65-2",
) -> float:
    summary = deltaE2000_summary(
        pred.clamp(0.0, 1.0),
        target.clamp(0.0, 1.0),
        whitepoint=whitepoint,
        percentiles=(95.0,),
    )
    return summary["p95"]


def edge_deltae2000_mean(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    whitepoint: Literal["D65-2", "D50-2"] = "D65-2",
    q: float = 0.85,
) -> float:
    stats = edge_deltaE2000(
        pred.clamp(0.0, 1.0),
        target.clamp(0.0, 1.0),
        whitepoint=whitepoint,
        q=q,
    )
    return stats["mean"]

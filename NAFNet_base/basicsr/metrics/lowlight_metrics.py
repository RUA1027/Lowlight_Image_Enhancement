"""Wrappers around project-level metrics for integration with BasicSR validation."""

from __future__ import annotations

from typing import Dict, Literal, Optional

import torch

from metrics.color_error import deltaE2000_summary, edge_deltaE2000
from metrics.lpips_metric import LPIPSEvaluator
from metrics.psnr import calculate_psnr
from metrics.ssim import calculate_ssim


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

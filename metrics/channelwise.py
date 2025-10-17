"""
Channel-wise image quality metrics (RGB-PSNR, CPSNR, optional RGB-SSIM).

For each RGB channel :math:`k`, we compute the mean squared error:

.. math::
    operatorname{MSE}_k = frac{1}{N H W} sum (x_k - y_k)^2,

and convert it to PSNR via

.. math::
    operatorname{PSNR}_k = 10 log_{10} left(frac{mathrm{MAX}^2}{operatorname{MSE}_k}\\right).

The CPSNR (Color PSNR) aggregates the channel MSEs before taking the logarithm:

.. math::
    operatorname{CMSE} = tfrac{1}{3} (operatorname{MSE}_R + operatorname{MSE}_G + \\operatorname{MSE}_B),\\qquad
    operatorname{CPSNR} = 10log_{10}left(frac{mathrm{MAX}^2}{operatorname{CMSE}}\\right).

These definitions follow standard practice in demosaicking and restoration
literature (e.g., Menon & Calvagno, Eq. 23) and require an explicit `data_range`
for floating-point images, in line with scikit-image's recommendations.
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple, TypedDict, Union

import torch
from torch import Tensor

class MetricMeta(TypedDict):
    domain: Literal["linear", "srgb"]
    data_range: float

class PSNRResult(TypedDict):
    R: Tensor
    G: Tensor
    B: Tensor
    mean: Tensor
    meta: MetricMeta

class CPSNRResult(TypedDict):
    cpsnr: Tensor
    meta: MetricMeta

class SSIMResult(TypedDict):
    R: Tensor
    G: Tensor
    B: Tensor
    mean: Tensor
    meta: MetricMeta

try:
    from .linear import ssim_linear
except ImportError:  # pragma: no cover - script execution outside package
    from linear import ssim_linear  # type: ignore

__all__ = ["rgb_psnr", "cpsnr_rgb", "rgb_ssim"]

_Reduction = Literal["mean", "sum", "none"]


def _ensure_rgb(pred: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Validate that tensors share shape/device and have 3 channels."""

    if not isinstance(pred, Tensor) or not isinstance(target, Tensor):
        raise TypeError("Expected torch.Tensor inputs for rgb metrics.")
    if pred.shape != target.shape:
        raise ValueError(
            f"`pred` and `target` must share identical shape, got {tuple(pred.shape)} vs {tuple(target.shape)}."
        )
    if pred.device != target.device:
        raise ValueError("`pred` and `target` must reside on the same device.")
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    elif pred.ndim != 4:
        raise ValueError(
            "Inputs must have 3 (C,H,W) or 4 (N,C,H,W) dimensions; "
            f"received tensor with shape {tuple(pred.shape)}."
        )
    if pred.shape[1] != 3:
        raise ValueError(
            f"Channel-wise metrics require exactly 3 channels (RGB). Received {pred.shape[1]}."
        )
    if not torch.isfinite(pred).all():
        raise ValueError("`pred` contains NaN or Inf values.")
    if not torch.isfinite(target).all():
        raise ValueError("`target` contains NaN or Inf values.")
    return pred.detach(), target.detach()


def _reduce(values: Tensor, reduction: _Reduction) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean(dim=0, keepdim=False)
    if reduction == "sum":
        return values.sum(dim=0, keepdim=False)
    raise ValueError(f"Unsupported reduction '{reduction}'. Choose from 'mean', 'sum', 'none'.")


def _clamp_opt(tensor: Tensor, clamp: bool | float, data_range: float) -> Tensor:
    if not clamp:
        return tensor
    hi = float(clamp) if isinstance(clamp, (int, float)) else float(data_range)
    return tensor.clamp(0.0, hi)


@torch.no_grad()
@torch.no_grad()
def rgb_psnr(
    pred: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
    reduction: _Reduction = "mean",
    domain: Literal["linear", "srgb"] = "linear",
    clamp: bool | float = False,
    meta: bool = False,
    eps: float = 1e-12,
) -> Union[Dict[str, Tensor], PSNRResult]:
    """Compute per-channel PSNR (higher is better) and optionally attach metadata.

    Args:
        pred: Predicted RGB tensor ``[N,3,H,W]`` or ``[3,H,W]``.
        target: Reference RGB tensor with identical shape.
        data_range: Peak value for PSNR; must be stated explicitly for floating-point data.
        reduction: Batch aggregation (``'mean'``, ``'sum'``, or ``'none'``).
        domain: Descriptive tag recorded alongside the metric (no colour conversion applied).
        clamp: Optional range guard (`True` clamps to ``[0, data_range]``).
        meta: When ``True``, include ``{'meta': {'domain': ..., 'data_range': ...}}`` in the result.
        eps: Numerical stabiliser guarding against division by zero.

    Returns:
        Dictionary with keys ``{"R", "G", "B", "mean"}`` (and ``"meta"`` when requested).
    """

    if data_range <= 0:
        raise ValueError(f"`data_range` must be positive; received {data_range}.")
    if eps <= 0:
        raise ValueError(f"`eps` must be positive; received {eps}.")

    pred_n, target_n = _ensure_rgb(pred, target)
    pred_n = _clamp_opt(pred_n.to(dtype=torch.float64), clamp, data_range)
    target_n = _clamp_opt(target_n.to(dtype=torch.float64), clamp, data_range)

    diff = pred_n - target_n
    mse = diff.pow(2).flatten(2).mean(dim=2)  # [N,3]
    max_val_sq = float(data_range) ** 2
    eps_tensor = torch.full_like(mse, eps)
    psnr = 10.0 * torch.log10(max_val_sq / torch.maximum(mse, eps_tensor))
    mean_psnr = psnr.mean(dim=1)

    base_result = {
        "R": _reduce(psnr[:, 0], reduction),
        "G": _reduce(psnr[:, 1], reduction),
        "B": _reduce(psnr[:, 2], reduction),
        "mean": _reduce(mean_psnr, reduction),
    }
    if not meta:
        return base_result
    return PSNRResult(
        **base_result,
        meta={"domain": domain, "data_range": float(data_range)}
    )
    return result

@torch.no_grad()
@torch.no_grad()
def cpsnr_rgb(
    pred: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
    reduction: _Reduction = "mean",
    domain: Literal["linear", "srgb"] = "linear",
    clamp: bool | float = False,
    meta: bool = False,
    eps: float = 1e-12,
) -> Union[Tensor, CPSNRResult]:
    """Compute colour PSNR (CPSNR) by averaging channel MSEs before taking the log.

    Args:
        pred: Predicted RGB tensor ``[N,3,H,W]`` or ``[3,H,W]``.
        target: Reference tensor with the same layout.
        data_range: Peak intensity for PSNR.
        reduction: Batch aggregation (`` + "mean" + , `` + "sum" + , `` + "none" + ).
        domain: Descriptive tag recorded alongside the metric.
        clamp: Optional range guard.
        meta: When ``True``, include cpsnr plus metadata in the return value.
        eps: Numerical stabiliser.

    Returns:
        CPSNR values (higher is better), or a dictionary when ``meta=True``.
    """

    if data_range <= 0:
        raise ValueError(f"`data_range` must be positive; received {data_range}.")
    if eps <= 0:
        raise ValueError(f"`eps` must be positive; received {eps}.")

    pred_n, target_n = _ensure_rgb(pred, target)
    pred_n = _clamp_opt(pred_n.to(dtype=torch.float64), clamp, data_range)
    target_n = _clamp_opt(target_n.to(dtype=torch.float64), clamp, data_range)

    diff = pred_n - target_n
    mse = diff.pow(2).flatten(2).mean(dim=2)  # [N,3]
    cmse = mse.mean(dim=1)

    max_val_sq = float(data_range) ** 2
    eps_tensor = torch.full_like(cmse, eps)
    cpsnr = 10.0 * torch.log10(max_val_sq / torch.maximum(cmse, eps_tensor))

    metric = _reduce(cpsnr, reduction)
    if not meta:
        return metric
    return CPSNRResult(
        cpsnr=metric,
        meta={"domain": domain, "data_range": float(data_range)}
    )

@torch.no_grad()
@torch.no_grad()
def rgb_ssim(
    pred: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    gaussian: bool = True,
    reduction: _Reduction = "mean",
    channel_aggregate: Literal["none", "mean"] = "none",
    padding: Literal["reflect", "replicate", "circular", "constant"] = "reflect",
    domain: Literal["linear", "srgb"] = "linear",
    meta: bool = False,
    eps: float = 1e-12,
) -> Union[Dict[str, Tensor], SSIMResult]:
    """Compute per-channel SSIM scores (higher is better) with optional metadata.

    Args:
        pred: Predicted RGB tensor ``[N,3,H,W]`` or ``[3,H,W]``.
        target: Reference tensor with matching shape.
        data_range: Peak intensity for SSIM constants.
        kernel_size/sigma/k1/k2/gaussian/padding: Classic SSIM parameters.
        reduction: Batch aggregation (``'mean'``, ``'sum'``, or ``'none'``).
        channel_aggregate: ``'none'`` keeps per-channel values, ``'mean'`` reports the same aggregated score for all channels.
        domain: Descriptive tag recorded alongside the metric.
        meta: When ``True``, include metadata with domain and data range.
        eps: Numerical stability constant.

    Returns:
        Dictionary containing SSIM per channel, the channel mean, and optional metadata.
    """

    pred_n, target_n = _ensure_rgb(pred, target)
    pred_n = pred_n.to(dtype=torch.float32)
    target_n = target_n.to(dtype=torch.float32)

    channel_scores: list[Tensor] = []
    for c in range(3):
        score = ssim_linear(
            pred_n[:, c : c + 1],
            target_n[:, c : c + 1],
            data_range=data_range,
            kernel_size=kernel_size,
            sigma=sigma,
            k1=k1,
            k2=k2,
            gaussian=gaussian,
            reduction="none",
            channel_aggregate="mean",
            padding=padding,
            eps=eps,
        )
        channel_scores.append(score.view(-1))

    stack = torch.stack(channel_scores, dim=1)
    mean_scores = stack.mean(dim=1)

    aggregated = _reduce(mean_scores, reduction)
    base_result = (
        {"R": aggregated, "G": aggregated, "B": aggregated, "mean": aggregated}
        if channel_aggregate == "mean"
        else {
            "R": _reduce(stack[:, 0], reduction),
            "G": _reduce(stack[:, 1], reduction),
            "B": _reduce(stack[:, 2], reduction),
            "mean": _reduce(mean_scores, reduction),
        }
    )

    if not meta:
        return base_result
    return SSIMResult(
        **base_result,
        meta={"domain": domain, "data_range": float(data_range)}
    )

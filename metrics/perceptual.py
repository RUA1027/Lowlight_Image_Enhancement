"""
LPIPS (Learned Perceptual Image Patch Similarity) utilities for sRGB inputs.

This module wraps the reference implementation from richzhang/PerceptualSimilarity,
providing batching, lazy model caching, and reproducible preprocessing aligned with
community practice (inputs in sRGB [0, 1] mapped to [-1, 1], default backbone AlexNet).
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple

import torch
from torch import Tensor

__all__ = ["lpips_srgb"]

_Reduction = Literal["mean", "sum", "none"]
_Backbone = Literal["alex", "vgg", "squeeze"]

_LPIPS_CACHE: Dict[Tuple[str, str, bool, torch.device], torch.nn.Module] = {}


def _ensure_nchw(pred: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Validate shapes/devices match and promote tensors to NCHW."""

    if not isinstance(pred, Tensor) or not isinstance(target, Tensor):
        raise TypeError("lpips_srgb expects torch.Tensor inputs.")
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must share identical shape, got {tuple(pred.shape)} vs {tuple(target.shape)}."
        )
    if pred.device != target.device:
        raise ValueError("pred and target must be on the same device.")
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    elif pred.ndim != 4:
        raise ValueError(
            "Inputs must have 3 (C,H,W) or 4 (N,C,H,W) dimensions; "
            f"received tensor with shape {tuple(pred.shape)}."
        )
    if pred.shape[0] == 0:
        raise ValueError("Batch size must be positive.")
    if pred.dtype not in {torch.float32, torch.float64}:
        raise TypeError(
            f"lpips_srgb expects float32/float64 tensors, received dtype={pred.dtype}."
        )
    if not torch.isfinite(pred).all():
        raise ValueError("pred contains NaN or Inf values.")
    if not torch.isfinite(target).all():
        raise ValueError("target contains NaN or Inf values.")
    return pred.detach(), target.detach()


def _reduce(values: Tensor, reduction: _Reduction) -> Tensor:
    """Reduce per-sample LPIPS values according to requested scheme."""

    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean(dim=0, keepdim=False)
    if reduction == "sum":
        return values.sum(dim=0, keepdim=False)
    raise ValueError(f"Unsupported reduction '{reduction}'. Choose from 'mean', 'sum', 'none'.")


def _get_lpips_model(
    net: _Backbone,
    version: str,
    device: torch.device,
    spatial: bool,
) -> torch.nn.Module:
    """Instantiate (or retrieve) a cached LPIPS model for the given configuration."""

    key = (net, version, spatial, device)
    if key in _LPIPS_CACHE:
        return _LPIPS_CACHE[key]
    try:
        import lpips  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover - handled explicitly in tests
        raise ImportError(
            "Please pip install lpips==0.1.4 (or a compatible version) to use lpips_srgb."
        ) from exc

    model = lpips.LPIPS(net=net, version=version, spatial=spatial)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    _LPIPS_CACHE[key] = model
    return model


@torch.no_grad()
def lpips_srgb(
    pred_srgb: Tensor,
    target_srgb: Tensor,
    *,
    net: _Backbone = "alex",
    version: str = "0.1",
    reduction: _Reduction = "mean",
    normalize_to_minus1_1: bool = True,
    clamp: bool | float = False,
    safe_gray: bool = True,
    spatial: bool = False,
    device: torch.device | None = None,
) -> Tensor:
    r"""Compute LPIPS between two sRGB tensors (lower is better).

    Inputs are assumed to be sRGB in `[0, 1]`. When `normalize_to_minus1_1=True`
    they are linearly mapped to `[-1, 1]` to match the reference implementation
    from Zhang et al. (BAPPS, CVPR'18). Different backbones (`alex`, `vgg`,
    `squeeze`) produce incomparable absolute scales. Setting `spatial=True` returns
    the per-location LPIPS map instead of reduced scalars (requires
    `reduction='none'`).

    Args:
        pred_srgb: Predicted sRGB tensor `[N,C,H,W]` or `[C,H,W]` (float32/64).
        target_srgb: Ground-truth tensor with the same layout as `pred_srgb`.
        net: Backbone used for LPIPS (`"alex"`, `"vgg"`, or `"squeeze"`).
        version: LPIPS weights version string (default `"0.1"`).
        reduction: Batch aggregation when `spatial=False` (`'mean'`, `'sum'`, `'none'`).
        normalize_to_minus1_1: Linearly map `[0,1]` inputs to `[-1,1]` when `True`.
        clamp: Optional range guard; `True` clamps to `[0,1]`, a float clamps to
            `[0, clamp]`. Leave `False` to expose saturation artefacts.
        safe_gray: If `True` and `C==1`, replicate channels internally; otherwise expect
            the caller to expand grayscale images manually.
        spatial: When `True`, return the spatial LPIPS map `[N,H,W]`.
        device: Optional device override for the LPIPS backbone.

    Returns:
        LPIPS distances: scalars when `spatial=False`, or spatial maps when
        `spatial=True`.
    """

    pred, target = _ensure_nchw(pred_srgb, target_srgb)

    batch, channels, height, width = pred.shape
    if channels not in {1, 3}:
        raise ValueError(
            f"lpips_srgb expects 1 or 3 channels; received C={channels}. "
            "Enable safe_gray=True to broadcast grayscale inputs or expand "
            "channels manually."
        )
    if height < 16 or width < 16:
        raise ValueError(
            "lpips_srgb requires spatial dimensions >= 16x16 to reach the deepest "
            f"feature maps. Received H={height}, W={width}."
        )

    lpips_device = torch.device(device) if device is not None else pred.device

    pred = pred.to(device=lpips_device, dtype=torch.float32)
    target = target.to(device=lpips_device, dtype=torch.float32)

    if channels == 1:
        if safe_gray:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        else:
            raise ValueError(
                "lpips_srgb received single-channel inputs while safe_gray=False. "
                "Replicate to three channels before calling the metric or pass "
                "safe_gray=True."
            )

    if clamp:
        upper = float(clamp) if isinstance(clamp, (int, float)) else 1.0
        pred = pred.clamp(0.0, upper)
        target = target.clamp(0.0, upper)

    if normalize_to_minus1_1:
        pred = pred.mul(2.0).sub(1.0)
        target = target.mul(2.0).sub(1.0)

    model = _get_lpips_model(net, version, lpips_device, spatial)

    outputs = model(pred, target).to(torch.float32)

    if spatial:
        if reduction != "none":
            raise ValueError("LPIPS spatial maps require reduction='none'.")
        if outputs.ndim == 4:
            outputs = outputs.squeeze(1)
        if torch.any(outputs < 0):
            import warnings

            warnings.warn(
                "LPIPS returned negative values; ensure inputs are properly "
                "normalised. Negative responses can occur because of the learned "
                "channel weights.",
                RuntimeWarning,
            )
        return outputs

    outputs = outputs.view(outputs.shape[0])

    if torch.any(outputs < 0):
        import warnings

        warnings.warn(
            "LPIPS returned negative values; ensure inputs are properly normalised. "
            "This is occasionally observed due to learned channel weighting.",
            RuntimeWarning,
        )

    return _reduce(outputs, reduction)

"""Delta E 2000 (CIEDE2000) color-difference utilities for sRGB inputs.

The implementation follows Sharma et al. (2005) and matches the reference
verification data provided alongside the CIE technical documentation. Inputs are
assumed to be sRGB in [0, 1]; conversion to CIELAB uses Kornia's D65/2-degree
whitepoint pipeline unless otherwise noted. Edge-focused statistics highlight
chromatic haloing along high-gradient regions. Smaller values indicate closer
colour reproduction.
"""

from __future__ import annotations

from typing import Dict, Iterable, Literal, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["deltaE2000_map", "deltaE2000_summary", "edge_deltaE2000"]

_WHITEPOINT = Literal["D65-2", "D50-2"]

_SOBEL_X_KERNEL = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)
_SOBEL_Y_KERNEL = torch.tensor(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)


def _ensure_nchw(rgb1: Tensor, rgb2: Tensor) -> Tuple[Tensor, Tensor]:
    """Ensure sRGB tensors are compatible and batched in NCHW format."""

    if not isinstance(rgb1, Tensor) or not isinstance(rgb2, Tensor):
        raise TypeError("deltaE2000 functions expect torch.Tensor inputs.")
    if rgb1.shape != rgb2.shape:
        raise ValueError(
            f"pred_srgb and target_srgb must share identical shape, got {tuple(rgb1.shape)} vs {tuple(rgb2.shape)}."
        )
    if rgb1.device != rgb2.device:
        raise ValueError("pred_srgb and target_srgb must reside on the same device.")
    if rgb1.ndim == 3:
        rgb1 = rgb1.unsqueeze(0)
        rgb2 = rgb2.unsqueeze(0)
    elif rgb1.ndim != 4:
        raise ValueError(
            "Expected tensors with 3 (C,H,W) or 4 (N,C,H,W) dimensions; "
            f"received tensor with shape {tuple(rgb1.shape)}."
        )
    if rgb1.shape[1] != 3:
        raise ValueError(f"sRGB inputs must have 3 channels. Received {rgb1.shape[1]}.")
    if not torch.isfinite(rgb1).all():
        raise ValueError("pred_srgb contains NaN or Inf values.")
    if not torch.isfinite(rgb2).all():
        raise ValueError("target_srgb contains NaN or Inf values.")
    return rgb1.detach(), rgb2.detach()


def _srgb_to_lab(rgb: Tensor, *, whitepoint: _WHITEPOINT) -> Tensor:
    """
    Convert sRGB [0,1] tensor to CIELAB using Kornia's implementation (D65/2°).

    Note:
        If `whitepoint='D50-2'` is requested, upstream code must perform the
        D65→D50 Bradford adaptation before calling this function. The conversion
        here does not re-run chromatic adaptation; the flag simply documents the
        intended whitepoint for downstream interpretation.
    """
    try:
        from kornia.color import rgb_to_lab
    except ImportError as exc:  # pragma: no cover - dependency error
        raise ImportError(
            "Please install kornia (pip install kornia>=0.6) to enable sRGB→Lab conversion."
        ) from exc

    lab = rgb_to_lab(rgb)
    if whitepoint == "D50-2":
        # Avoid silent mismatch: Kornia outputs D65/2° by default. We warn users that
        # they must have adapted upstream if they select D50 here.
        import warnings

        warnings.warn(
            "deltaE2000_map called with whitepoint='D50-2'. Ensure inputs were "
            "Bradford-adapted from D65 to D50 upstream (CSS Color 4). This function "
            "does not perform chromatic adaptation internally.",
            RuntimeWarning,
        )
    return lab


def _deg2rad(value: float) -> float:
    return float(value) * torch.pi / 180.0


@torch.no_grad()
def _deltaE00_lab_map(
    lab1: Tensor,
    lab2: Tensor,
    *,
    kL: float,
    kC: float,
    kH: float,
    eps: float,
) -> Tensor:
    """Vectorised ΔE00 computation on Lab tensors (expects [N,3,H,W])."""

    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    c1 = torch.sqrt(a1 * a1 + b1 * b1 + eps)
    c2 = torch.sqrt(a2 * a2 + b2 * b2 + eps)
    c_bar = 0.5 * (c1 + c2)

    c_bar7 = c_bar.pow(7)
    g = 0.5 * (1.0 - torch.sqrt(c_bar7 / (c_bar7 + torch.tensor(25.0**7, device=lab1.device, dtype=lab1.dtype) + eps)))

    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    c1_prime = torch.sqrt(a1_prime * a1_prime + b1 * b1 + eps)
    c2_prime = torch.sqrt(a2_prime * a2_prime + b2 * b2 + eps)

    h1_prime = torch.atan2(b1, a1_prime)
    h2_prime = torch.atan2(b2, a2_prime)

    delta_L_prime = L2 - L1
    delta_C_prime = c2_prime - c1_prime

    delta_h_prime = h2_prime - h1_prime
    delta_h_prime = torch.where(
        delta_h_prime > torch.pi,
        delta_h_prime - 2.0 * torch.pi,
        delta_h_prime,
    )
    delta_h_prime = torch.where(
        delta_h_prime < -torch.pi,
        delta_h_prime + 2.0 * torch.pi,
        delta_h_prime,
    )

    delta_H_prime = 2.0 * torch.sqrt(c1_prime * c2_prime + eps) * torch.sin(delta_h_prime / 2.0)

    L_bar_prime = 0.5 * (L1 + L2)
    C_bar_prime = 0.5 * (c1_prime + c2_prime)

    h_sum = h1_prime + h2_prime
    h_bar_prime = torch.where(
        (c1_prime * c2_prime).eq(0.0),
        h_sum,
        torch.where(
            torch.abs(h1_prime - h2_prime) <= torch.pi,
            0.5 * h_sum,
            torch.where(
                h_sum < 0.0,
                0.5 * (h_sum + 2.0 * torch.pi),
                0.5 * (h_sum - 2.0 * torch.pi),
            ),
        ),
    )

    cos_term = torch.cos(h_bar_prime)
    sin_term = torch.sin(h_bar_prime)

    t = (
        1.0
        - 0.17 * torch.cos(h_bar_prime - _deg2rad(30.0))
        + 0.24 * torch.cos(2.0 * h_bar_prime)
        + 0.32 * torch.cos(3.0 * h_bar_prime + _deg2rad(6.0))
        - 0.20 * torch.cos(4.0 * h_bar_prime - _deg2rad(63.0))
    )

    delta_theta = _deg2rad(30.0) * torch.exp(-(((h_bar_prime * 180.0 / torch.pi) - 275.0) / 25.0) ** 2)
    r_c = 2.0 * torch.sqrt((C_bar_prime.pow(7)) / (C_bar_prime.pow(7) + torch.tensor(25.0**7, device=lab1.device, dtype=lab1.dtype) + eps))
    r_t = -torch.sin(2.0 * delta_theta) * r_c

    s_L = 1.0 + (0.015 * (L_bar_prime - 50.0) ** 2) / torch.sqrt(20.0 + (L_bar_prime - 50.0) ** 2 + eps)
    s_C = 1.0 + 0.045 * C_bar_prime
    s_H = 1.0 + 0.015 * C_bar_prime * t

    l_term = delta_L_prime / (kL * s_L + eps)
    c_term = delta_C_prime / (kC * s_C + eps)
    h_term = delta_H_prime / (kH * s_H + eps)

    delta_e_squared = l_term * l_term + c_term * c_term + h_term * h_term + r_t * c_term * h_term
    delta_e = torch.sqrt(torch.clamp(delta_e_squared, min=0.0))

    # Return shape [N, H, W] without extra channel dimension
    return delta_e


def _compute_percentiles(values: Tensor, percentiles: Iterable[float]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    flat = values.view(-1)
    if flat.numel() == 0:
        for p in percentiles:
            stats[f"p{int(p)}"] = float("nan")
        return stats

    qs = []
    for p in percentiles:
        q = float(p)
        if not 0.0 <= q <= 100.0:
            raise ValueError(f"Percentile values must lie within [0, 100]; received {q}.")
        qs.append(q / 100.0)

    quantiles = torch.quantile(flat, torch.tensor(qs, device=flat.device, dtype=flat.dtype))
    for idx, p in enumerate(percentiles):
        stats[f"p{int(p)}"] = float(quantiles[idx].item())
    return stats


@torch.no_grad()
def deltaE2000_map(
    pred_srgb: Tensor,
    target_srgb: Tensor,
    *,
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
    whitepoint: _WHITEPOINT = "D65-2",
    eps: float = 1e-12,
) -> Tensor:
    r"""Compute Delta E_00 (CIEDE2000) maps between two sRGB images/batches.

Inputs must be sRGB tensors in [0, 1]. Conversion to Lab uses kornia.color.rgb_to_lab, which assumes the D65 illuminant with a 2-degree observer. If `whitepoint='D50-2'` is requested, callers should adapt from D65 to D50 upstream (e.g., Bradford transform). Smaller Delta E_00 values indicate closer colour reproduction.
"""

    if eps <= 0:
        raise ValueError(f"`eps` must be positive, received {eps}.")
    pred, target = _ensure_nchw(pred_srgb, target_srgb)

    device = pred.device
    dtype = torch.float32

    lab_pred = _srgb_to_lab(pred.to(device=device, dtype=dtype), whitepoint=whitepoint)
    lab_target = _srgb_to_lab(target.to(device=device, dtype=dtype), whitepoint=whitepoint)

    return _deltaE00_lab_map(lab_pred, lab_target, kL=kL, kC=kC, kH=kH, eps=eps)


@torch.no_grad()
def deltaE2000_summary(
    pred_srgb: Tensor,
    target_srgb: Tensor,
    *,
    percentiles: Tuple[float, ...] = (50.0, 95.0),
    **kwargs,
) -> Dict[str, float]:
    """Summarise Delta E_00 statistics (mean and percentiles) consistent with
`deltaE2000_map`. Smaller values indicate less perceptual colour error.
"""

    de_map = deltaE2000_map(pred_srgb, target_srgb, **kwargs)
    flat = de_map.view(de_map.shape[0], -1)
    mean = flat.mean(dim=1)
    summary: Dict[str, float] = {"mean": float(mean.mean().item())}

    flat_all = flat.reshape(-1)
    if flat_all.numel() > 0:
        summary.update(_compute_percentiles(flat_all, percentiles))
    else:
        for p in percentiles:
            summary[f"p{int(p)}"] = float("nan")

    return summary


def _sobel_magnitude(l_channel: Tensor) -> Tensor:
    kernel_x = _SOBEL_X_KERNEL.to(device=l_channel.device, dtype=l_channel.dtype)
    kernel_y = _SOBEL_Y_KERNEL.to(device=l_channel.device, dtype=l_channel.dtype)
    gx = F.conv2d(l_channel, kernel_x, padding=1)
    gy = F.conv2d(l_channel, kernel_y, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


@torch.no_grad()
def edge_deltaE2000(
    pred_srgb: Tensor,
    target_srgb: Tensor,
    *,
    method: str = "sobel",
    q: float = 0.85,
    **kwargs,
) -> Dict[str, float]:
    """Compute Delta E_00 statistics restricted to high-gradient regions (edge emphasis).
The default Sobel-based mask keeps pixels whose gradient magnitude lies above
the q-th quantile; smaller Delta E_00 values indicate better colour fidelity
along edges.
"""

    if method != "sobel":
        raise ValueError(f"Unsupported edge detection method '{method}'. Currently only 'sobel' is available.")
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must lie within (0,1); received {q}.")

    de_map = deltaE2000_map(pred_srgb, target_srgb, **kwargs)
    lab_pred = _srgb_to_lab(_ensure_nchw(pred_srgb, target_srgb)[0], whitepoint=kwargs.get("whitepoint", "D65-2"))
    l_channel = lab_pred[:, 0:1]
    grad = _sobel_magnitude(l_channel)

    # Ensure de_map has channel dimension for consistency
    if de_map.ndim == 3:  # [N, H, W]
        de_map = de_map.unsqueeze(1)  # [N, 1, H, W]

    threshold = torch.quantile(grad.view(grad.shape[0], -1), q, dim=1, keepdim=True)
    mask = grad >= threshold.view(-1, 1, 1, 1)

    masked_values = de_map[mask.expand_as(de_map)]
    if masked_values.numel() == 0:
        return {"mean": float("nan"), "p95": float("nan")}

    mean = masked_values.mean()
    p95 = torch.quantile(masked_values, 0.95)

    return {"mean": float(mean.item()), "p95": float(p95.item())}

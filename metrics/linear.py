"""
Linear-domain PSNR and SSIM metrics delivered as lightweight functional utilities.

The implementations follow the academic definitions referenced in ITU-R BT.601
and Wang et al. (2004), using explicit control over data range, window shape and
padding so that results stay reproducible across datasets and hardware.
"""

# type: ignore  # Suppress Pylance false positives for torch attributes

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["psnr_linear", "ssim_linear"]

_Reduction = Literal["mean", "sum", "none"]
_ChannelAggregate = Literal["mean", "none"]
_PaddingMode = Literal["reflect", "replicate", "circular", "constant"]


def _ensure_nchw(pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Validate tensor compatibility and promote to NCHW."""

    if not isinstance(pred, Tensor) or not isinstance(target, Tensor):
        raise TypeError("psnr_linear/ssim_linear expect torch.Tensor inputs.")
    if pred.dtype not in {torch.float32, torch.float64}:
        raise TypeError(
            f"Expected pred dtype float32/float64, received {pred.dtype}."
        )
    if target.dtype != pred.dtype:
        raise TypeError("pred and target must share the same dtype.")
    if pred.device != target.device:
        raise ValueError("pred and target must live on the same device.")
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must share identical shape, got {pred.shape} vs {target.shape}."
        )
    if not torch.isfinite(pred).all():
        raise ValueError("pred contains NaN or Inf values.")
    if not torch.isfinite(target).all():
        raise ValueError("target contains NaN or Inf values.")

    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    elif pred.ndim != 4:
        raise ValueError(
            "Inputs must have 3 (C,H,W) or 4 (N,C,H,W) dimensions; "
            f"received tensor with shape {tuple(pred.shape)}."
        )

    if pred.shape[0] == 0:
        raise ValueError("Batch dimension cannot be zero.")
    if pred.shape[1] == 0:
        raise ValueError("Channel dimension cannot be zero.")
    if pred.shape[2] == 0 or pred.shape[3] == 0:
        raise ValueError("Spatial dimensions must be strictly positive.")

    return pred.detach(), target.detach()


def _reduce(values: Tensor, reduction: _Reduction) -> Tensor:
    """Apply reduction across the batch dimension."""

    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean(dim=0)
    if reduction == "sum":
        return values.sum(dim=0)
    raise ValueError(
        f"Unsupported reduction='{reduction}'. Expected 'mean', 'sum', or 'none'."
    )


@lru_cache(maxsize=None)
def _kernel_cache_base(kernel_size: int, sigma: float, gaussian: bool) -> Tensor:
    """Return a cached 2D window on CPU (float64)."""

    device = torch.device("cpu")
    dtype = torch.float64
    if gaussian:
        if sigma <= 0:
            raise ValueError("sigma must be positive when gaussian=True.")
        coords = torch.arange(kernel_size, dtype=dtype, device=device)
        coords = coords - (kernel_size - 1) / 2.0
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    else:
        kernel_2d = torch.ones((kernel_size, kernel_size), dtype=dtype, device=device)
    return kernel_2d


def _build_kernel(
    kernel_size: int,
    sigma: float,
    gaussian: bool,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Create a normalized 2D kernel (Gaussian or uniform) with shape [1,1,k,k]."""

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be a positive odd integer; received "
            f"{kernel_size}."
        )
    kernel_size = int(kernel_size)

    cache_sigma = float(sigma if gaussian else 0.0)
    base = _kernel_cache_base(kernel_size, cache_sigma, bool(gaussian))
    kernel = base.to(device=device, dtype=dtype)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def _pad(
    tensor: Tensor,
    pad: int,
    mode: _PaddingMode,
) -> Tensor:
    """Apply symmetric padding before convolution."""

    if pad == 0:
        return tensor

    if mode == "constant":
        return F.pad(tensor, (pad, pad, pad, pad), mode=mode, value=0.0)
    return F.pad(tensor, (pad, pad, pad, pad), mode=mode)


@torch.no_grad()
def psnr_linear(
    pred: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
    reduction: _Reduction = "mean",
    clamp: bool | float = False,
    eps: float = 1e-12,
) -> Tensor:
    r"""Linear-domain Peak Signal-to-Noise Ratio (PSNR).

    For linear-domain tensors :math:`\mathbf{x}` (prediction) and
    :math:`\mathbf{y}` (reference),

    .. math::
        \mathrm{PSNR} = 10 \log_{10}\left(\frac{L^2}{\mathrm{MSE}}\right),
        \qquad
        \mathrm{MSE} = \frac{1}{NCHW}\sum (\mathbf{x} - \mathbf{y})^2,

    where :math:`L` equals `data_range`, the peak linear intensity. No gamma or
    colour-space conversion is performed inside this function.

    Args:
        pred: Predicted tensor `[N,C,H,W]` or `[C,H,W]` in the linear domain.
        target: Ground-truth tensor matching the shape and dtype of `pred`.
        data_range: Positive scalar defining the peak intensity (e.g. `1.0`,
            `255.0`, `4095.0`).
        reduction: Batch aggregation mode (`'mean'`, `'sum'`, or `'none'`).
        clamp: Optional range guard. `True` clamps both inputs to
            `[0, data_range]`; a float clamps to `[0, clamp]`; `False` disables
            clamping.
        eps: Numerical floor applied to the MSE before taking the logarithm.

    Returns:
        Tensor of PSNR scores in decibels. Shape depends on `reduction`.
    """

    if data_range <= 0:
        raise ValueError(f"`data_range` must be positive, received {data_range}.")
    if eps <= 0:
        raise ValueError(f"`eps` must be positive, received {eps}.")

    pred, target = _ensure_nchw(pred, target)

    clamp_range: tuple[float, float] | None
    clamp_range = None
    if isinstance(clamp, bool):
        if clamp:
            clamp_range = (0.0, float(data_range))
    else:
        clamp_range = (0.0, float(clamp))

    if clamp_range is not None:
        low, high = clamp_range
        pred = pred.clamp(low, high)
        target = target.clamp(low, high)

    diff = (pred - target).to(torch.float64)
    mse = diff.pow(2).flatten(1).mean(dim=1)
    floor = torch.full_like(mse, fill_value=eps)
    safe_mse = torch.maximum(mse, floor)
    psnr = 10.0 * torch.log10((float(data_range) ** 2) / safe_mse)
    zero_mask = mse <= eps
    if zero_mask.any():
        psnr = torch.where(
            zero_mask,
            torch.full_like(psnr, float("inf")),
            psnr,
        )

    return _reduce(psnr, reduction)


@torch.no_grad()
def ssim_linear(
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
    channel_aggregate: _ChannelAggregate = "mean",
    padding: _PaddingMode = "reflect",
    eps: float = 1e-12,
) -> Tensor:
    r"""Structural Similarity (SSIM) in the linear intensity domain.

    For each window, SSIM is computed as

    .. math::
        \mathrm{SSIM} =
            \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}
                 { (\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2) },

    with :math:`\mu` and :math:`\sigma` estimated via a sliding window. We adopt
    the classic Wang et al. (2004) configuration: `K1=0.01`, `K2=0.03`, and an
    11x11 Gaussian window with sigma approximately 1.5. The default 'reflect'
    padding reduces boundary bias compared with zero padding.

    Args:
        pred: Predicted tensor `[N,C,H,W]` or `[C,H,W]` in linear space.
        target: Ground-truth tensor matching the shape/dtype of `pred`.
        data_range: Positive intensity span for linear data.
        kernel_size: Odd window size (default 11).
        sigma: Gaussian standard deviation (ignored if `gaussian=False`).
        k1: Constant used to derive `C1 = (k1 * data_range)^2`.
        k2: Constant used to derive `C2 = (k2 * data_range)^2`.
        gaussian: Use Gaussian window when `True`, uniform otherwise.
        reduction: Batch aggregation (`'mean'`, `'sum'`, or `'none'`).
        channel_aggregate: `'mean'` averages channels, `'none'` returns per-channel scores.
        padding: Border handling (`'reflect'` recommended).
        eps: Numerical stability term for denominators.

    Returns:
        Tensor of SSIM scores (roughly `[-1, 1]`), shaped according to the
        chosen `reduction` and `channel_aggregate`.
    """

    if data_range <= 0:
        raise ValueError(f"`data_range` must be positive, received {data_range}.")
    if eps <= 0:
        raise ValueError(f"`eps` must be positive, received {eps}.")
    if k1 < 0 or k2 < 0:
        raise ValueError("k1 and k2 must be non-negative.")
    if channel_aggregate not in {"mean", "none"}:
        raise ValueError(
            f"channel_aggregate must be 'mean' or 'none', received {channel_aggregate}."
        )

    pred, target = _ensure_nchw(pred, target)

    n, c, h, w = pred.shape
    if h < kernel_size or w < kernel_size:
        raise ValueError(
            "Spatial dimensions must be >= kernel_size. "
            f"Got H={h}, W={w}, kernel_size={kernel_size}."
        )

    kernel = _build_kernel(
        kernel_size,
        sigma,
        gaussian,
        dtype=pred.dtype,
        device=pred.device,
    )
    kernel = kernel.repeat(c, 1, 1, 1)

    pad = kernel_size // 2
    pred_padded = _pad(pred, pad, padding)
    target_padded = _pad(target, pad, padding)

    mu_x = F.conv2d(pred_padded, kernel, groups=c)
    mu_y = F.conv2d(target_padded, kernel, groups=c)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred_padded * pred_padded, kernel, groups=c) - mu_x2
    sigma_y2 = F.conv2d(target_padded * target_padded, kernel, groups=c) - mu_y2
    sigma_xy = F.conv2d(pred_padded * target_padded, kernel, groups=c) - mu_xy

    # Clamp variances to avoid numerical instability from negative values
    # due to floating-point precision errors
    sigma_x2 = torch.clamp(sigma_x2, min=0.0)
    sigma_y2 = torch.clamp(sigma_y2, min=0.0)

    c1 = (k1 * float(data_range)) ** 2
    c2 = (k2 * float(data_range)) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + eps)

    ssim_per_channel = ssim_map.flatten(2).mean(dim=2)

    if channel_aggregate == "mean":
        ssim_per_image = ssim_per_channel.mean(dim=1)
    else:
        ssim_per_image = ssim_per_channel

    return _reduce(ssim_per_image, reduction)

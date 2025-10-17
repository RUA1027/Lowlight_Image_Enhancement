"""
Physics-consistency metrics under a linear, shift-invariant imaging model.

We assume the short-exposure observation :math:`A` relates to the restored long
exposure :math:`hat{B}` via a convolution with the point spread function (PSF)
followed by an exposure gain :math:`rho` and noise:

.. math::
    A approx operatorname{clip}bigl( rho cdot (K * hat{B}) bigr) + n,

where :math:`K` is the system PSF (capturing pixel crosstalk/MTF) and :math:`n`
models Poisson Gaussian noise as observed in low-light pipelines such as SID
([Chen et al. CVPR'18](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Learning_to_See_CVPR_2018_paper.pdf)).
Lower scores indicate that the forward-simulated observation agrees with the
measured short exposure, providing direct evidence that pre-compensation is
physically faithful.
"""

from __future__ import annotations

from typing import Literal, Tuple

import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["phys_cons_raw", "phys_cons_srgb"]

_Reduction = Literal["mean", "sum", "none"]
_Padding = Literal["reflect", "replicate", "zeros"]
_Crop = Literal["valid", "same"]
_Robust = Literal["none", "charbonnier"]


def _ensure_nchw(pred: Tensor, obs: Tensor) -> Tuple[Tensor, Tensor]:
    """Validate tensor compatibility and promote inputs to NCHW layout."""

    if not isinstance(pred, Tensor) or not isinstance(obs, Tensor):
        raise TypeError("phys_consistency metrics expect torch.Tensor inputs.")
    if pred.shape != obs.shape:
        raise ValueError(
            f"`pred` and `obs` must share identical shape, got {tuple(pred.shape)} vs {tuple(obs.shape)}."
        )
    if pred.device != obs.device:
        raise ValueError("`pred` and `obs` must reside on the same device.")
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        obs = obs.unsqueeze(0)
    elif pred.ndim != 4:
        raise ValueError(
            "Inputs must have 3 (C,H,W) or 4 (N,C,H,W) dimensions; "
            f"received tensor with shape {tuple(pred.shape)}."
        )
    if pred.shape[1] == 0:
        raise ValueError("Channel dimension must be positive.")
    if not torch.isfinite(pred).all():
        raise ValueError("`pred` contains NaN or Inf values.")
    if not torch.isfinite(obs).all():
        raise ValueError("`obs` contains NaN or Inf values.")
    return pred.detach(), obs.detach()


def _reduce(values: Tensor, reduction: _Reduction) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean(dim=0, keepdim=False)
    if reduction == "sum":
        return values.sum(dim=0, keepdim=False)
    raise ValueError(f"Unsupported reduction '{reduction}'. Choose from 'mean', 'sum', 'none'.")


def _prepare_psf(
    psf: Tensor,
    *,
    in_channels: int,
    out_channels: int,
    device: torch.device,
    dtype: torch.dtype,
    normalize: bool,
    enforce_nonnegative: bool,
    eps: float,
) -> Tensor:
    """Validate PSF shape and optionally normalise energy per output channel."""

    if not isinstance(psf, Tensor):
        raise TypeError("`psf` must be a torch.Tensor.")
    if psf.ndim == 2:
        psf = psf.unsqueeze(0).unsqueeze(0)
    if psf.ndim != 4:
        raise ValueError(
            f"`psf` must have shape [C_out, C_in, kh, kw]; received tensor with shape {tuple(psf.shape)}."
        )
    c_out, c_in, kh, kw = psf.shape
    if c_out != out_channels:
        raise ValueError(
            f"PSF output channels ({c_out}) must match observation channels ({out_channels})."
        )
    if c_in != in_channels:
        raise ValueError(
            f"PSF input channels ({c_in}) must match prediction channels ({in_channels})."
        )
    if kh < 1 or kw < 1:
        raise ValueError("PSF kernel height/width must be >= 1.")
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(
            "PSF kernels must have odd spatial dimensions to avoid half-pixel shifts. "
            "Please supply odd-sized kernels (e.g., 3/5/7)."
        )

    psf = psf.to(device=device, dtype=dtype)
    if enforce_nonnegative:
        psf = psf.clamp_min(0)
    if normalize:
        sums = psf.view(c_out, -1).sum(dim=1)
        zero_mask = sums.abs() < eps
        if zero_mask.any():
            warnings.warn(
                "PSF channel sums near zero detected during normalisation; "
                "clamping to preserve stability.",
                RuntimeWarning,
            )
        denom = torch.where(zero_mask, torch.ones_like(sums), sums)
        psf = psf / denom.view(c_out, 1, 1, 1)
    return psf


def _apply_psf(x: Tensor, psf: Tensor, padding: _Padding) -> Tensor:
    """Convolve batched tensor with PSF using the specified padding scheme."""

    kh, kw = psf.shape[-2:]
    pad = (kw // 2, kw // 2, kh // 2, kh // 2)
    if padding == "reflect":
        x_pad = F.pad(x, pad, mode="reflect")
        return F.conv2d(x_pad, psf)
    if padding == "replicate":
        x_pad = F.pad(x, pad, mode="replicate")
        return F.conv2d(x_pad, psf)
    if padding == "zeros":
        return F.conv2d(x, psf, padding=(kh // 2, kw // 2))
    raise ValueError(f"Unsupported padding mode '{padding}'.")


def _crop_valid(y_hat: Tensor, obs: Tensor, kh: int, kw: int) -> Tuple[Tensor, Tensor]:
    """Remove boundary margins induced by convolution (valid region)."""

    pad_h = kh // 2
    pad_w = kw // 2
    if pad_h > 0:
        y_hat = y_hat[..., pad_h:-pad_h, :]
        obs = obs[..., pad_h:-pad_h, :]
    if pad_w > 0:
        y_hat = y_hat[..., :, pad_w:-pad_w]
        obs = obs[..., :, pad_w:-pad_w]
    return y_hat, obs


def _expand_exposure(expo_ratio: float | Tensor, ref: Tensor) -> Tensor:
    """Broadcast exposure ratio to match reference tensor."""

    if torch.is_tensor(expo_ratio):
        ratio = expo_ratio.to(device=ref.device, dtype=ref.dtype)
    else:
        ratio = torch.tensor(float(expo_ratio), device=ref.device, dtype=ref.dtype)
    if ratio.ndim == 0:
        ratio = ratio.view(1, 1, 1, 1).expand(ref.shape[0], 1, 1, 1)
    elif ratio.ndim == 1:
        if ratio.shape[0] != ref.shape[0]:
            raise ValueError(
                f"Exposure ratio length ({ratio.shape[0]}) must match batch size ({ref.shape[0]})."
            )
        ratio = ratio.view(ref.shape[0], 1, 1, 1)
    elif ratio.ndim == 4:
        if ratio.shape[0] != ref.shape[0]:
            raise ValueError(
                f"Exposure ratio batch dimension ({ratio.shape[0]}) must match batch size ({ref.shape[0]})."
            )
        if ratio.shape[1] == 1 and ref.shape[1] > 1:
            ratio = ratio.expand(ref.shape[0], ref.shape[1], ratio.shape[2], ratio.shape[3])
        elif ratio.shape[1] not in (1, ref.shape[1]):
            raise ValueError(
                f"Exposure ratio channel dimension ({ratio.shape[1]}) incompatible with data channels ({ref.shape[1]})."
            )
    else:
        raise ValueError(
            "Exposure ratio must be scalar, [N], or [N,1,H,W]/[N,C,H,W] for broadcasting."
        )
    return ratio


def _phys_cons_core(
    pred: Tensor,
    obs: Tensor,
    *,
    psf: Tensor,
    expo_ratio: float | Tensor,
    reduction: _Reduction,
    padding: _Padding,
    normalize_psf: bool,
    enforce_nonnegative: bool,
    crop: _Crop,
    robust: _Robust,
    return_map: bool,
    clamp_range: Tuple[float, float] | None,
    eps: float,
) -> Tensor | Tuple[Tensor, Tensor]:
    if eps <= 0:
        raise ValueError(f"`eps` must be positive, received {eps}.")
    if robust not in {"none", "charbonnier"}:
        raise ValueError(f"Unsupported robust loss '{robust}'.")
    if crop not in {"valid", "same"}:
        raise ValueError(f"Unsupported crop mode '{crop}'.")
    if padding not in {"reflect", "replicate", "zeros"}:
        raise ValueError(f"Unsupported padding mode '{padding}'.")

    psf_prepared = _prepare_psf(
        psf,
        in_channels=pred.shape[1],
        out_channels=obs.shape[1],
        device=pred.device,
        dtype=pred.dtype,
        normalize=normalize_psf,
        enforce_nonnegative=enforce_nonnegative,
        eps=eps,
    )

    y_hat = _apply_psf(pred, psf_prepared, padding)
    ratio = _expand_exposure(expo_ratio, y_hat)
    if ratio.shape[1] == 1 and y_hat.shape[1] != 1:
        ratio = ratio.expand(y_hat.shape[0], y_hat.shape[1], ratio.shape[2], ratio.shape[3])
    y_hat = y_hat * ratio

    if clamp_range is not None:
        y_hat = y_hat.clamp(clamp_range[0], clamp_range[1])

    obs_comp = obs
    if crop == "valid":
        kh, kw = psf_prepared.shape[-2:]
        y_hat, obs_comp = _crop_valid(y_hat, obs_comp, kh, kw)

    diff = y_hat - obs_comp
    abs_diff = diff.abs()
    if robust == "charbonnier":
        loss_map = torch.sqrt(diff * diff + eps * eps)
    else:
        loss_map = abs_diff

    per_sample = loss_map.flatten(1).mean(dim=1)
    metric = _reduce(per_sample, reduction)

    if return_map:
        return metric, abs_diff
    return metric

@torch.no_grad()
@torch.no_grad()
@torch.no_grad()
def phys_cons_raw(
    pred_linear: Tensor,
    obs_short_linear: Tensor,
    psf: Tensor,
    expo_ratio: float | Tensor,
    *,
    reduction: _Reduction = "mean",
    padding: _Padding = "reflect",
    normalize_psf: bool = True,
    enforce_nonnegative: bool = False,
    crop: _Crop = "valid",
    robust: _Robust = "none",
    return_map: bool = False,
    eps: float = 1e-12,
) -> Tensor | Tuple[Tensor, Tensor]:
    """Physics-consistency error in the RAW / linear domain (lower is better).

    Simulates the forward model rho * (K * B_hat) with the supplied PSF and
    exposure ratio, then compares the synthesised observation against the short
    exposure via an L1 or Charbonnier discrepancy. The PSF models a linear,
    shift-invariant system; enabling ``enforce_nonnegative`` clamps negative PSF
    lobes before normalisation to keep the interpretation physically meaningful.

    Args:
        pred_linear: Restored long-exposure estimate in linear space.
        obs_short_linear: Registered short-exposure observation.
        psf: Crosstalk PSF ``[C_out, C_in, kh, kw]`` (may couple colour channels).
        expo_ratio: Exposure gain rho; scalar or broadcastable tensor.
        reduction: Batch aggregation (``'mean'``, ``'sum'``, or ``'none'``).
        padding: Boundary handling (``'reflect'`` recommended to curb artefacts).
        normalize_psf: Normalise each output-channel kernel so sum K = 1.
        enforce_nonnegative: Clamp PSF entries below zero before normalisation.
        crop: ``'valid'`` removes margins introduced by convolution; ``'same'`` keeps size.
        robust: ``'none'`` yields MAE; ``'charbonnier'`` applies a smooth robust penalty.
        return_map: When ``True``, also return the absolute residual map ``|PSF(B_hat) - A|``.
        eps: Numerical stabiliser for Charbonnier and normalisation.

    Returns:
        Physics-consistency scores (and optionally residual maps when ``return_map=True``).
    """

    pred, obs = _ensure_nchw(pred_linear, obs_short_linear)
    pred = pred.to(dtype=torch.float32)
    obs = obs.to(dtype=torch.float32)

    return _phys_cons_core(
        pred,
        obs,
        psf=psf,
        expo_ratio=expo_ratio,
        reduction=reduction,
        padding=padding,
        normalize_psf=normalize_psf,
        enforce_nonnegative=enforce_nonnegative,
        crop=crop,
        robust=robust,
        return_map=return_map,
        clamp_range=None,
        eps=eps,
    )

@torch.no_grad()
@torch.no_grad()
def phys_cons_srgb(
    pred_srgb: Tensor,
    obs_short_srgb: Tensor,
    psf: Tensor,
    expo_ratio: float | Tensor = 1.0,
    *,
    reduction: _Reduction = "mean",
    padding: _Padding = "reflect",
    normalize_psf: bool = True,
    enforce_nonnegative: bool = False,
    crop: _Crop = "valid",
    robust: _Robust = "none",
    clamp01: bool = True,
    return_map: bool = False,
    eps: float = 1e-12,
) -> Tensor | Tuple[Tensor, Tensor]:
    """Physics-consistency error computed in the sRGB domain (auxiliary evidence).

    Mirrors :func:`phys_cons_raw` but expects sRGB inputs in [0, 1]. The optional
    ``clamp01`` flag clips the synthesised observation after applying the PSF and
    exposure scaling. The RAW metric should remain the primary indicator of
    physical fidelity; this variant is provided for qualitative checks in display
    space.
    """

    pred, obs = _ensure_nchw(pred_srgb, obs_short_srgb)
    pred = pred.to(dtype=torch.float32)
    obs = obs.to(dtype=torch.float32)

    clamp_range = (0.0, 1.0) if clamp01 else None

    return _phys_cons_core(
        pred,
        obs,
        psf=psf,
        expo_ratio=expo_ratio,
        reduction=reduction,
        padding=padding,
        normalize_psf=normalize_psf,
        enforce_nonnegative=enforce_nonnegative,
        crop=crop,
        robust=robust,
        return_map=return_map,
        clamp_range=clamp_range,
        eps=eps,
    )

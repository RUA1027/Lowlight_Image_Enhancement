"""SSIM metric utilities (model-agnostic, evaluation oriented).

This module implements a reusable, neutral SSIM evaluator that follows the
single-scale, Gaussian-window specification from Wang et al. (2004):

SSIM(x, y) = ((2 mu_x mu_y + C1) (2 sigma_xy + C2)) / ((mu_x^2 + mu_y^2 + C1)
             (sigma_x^2 + sigma_y^2 + C2)),

where mu and sigma are local means/variances computed with a Gaussian kernel
of size 11x11 and sigma=1.5, and C1=(K1*L)^2, C2=(K2*L)^2 with K1=0.01,
K2=0.03. The evaluator enforces consistent input assumptions so that control
and experiment groups can be compared under the exact same measurement
protocol. Key guarantees:

* Inputs are accepted in RGB or grayscale, with automatic batch handling.
* Pixel range is auto-inferred among [0,1], [-1,1], or [0,255] unless the
  caller provides an explicit ``data_range``.
* Spatial dimensions must match unless an explicit resize policy is requested.
* All outputs include per-image SSIM scores plus summary statistics
  (mean/std/p50/p95) together with the configuration that influences the
  reported values (kernel_size, sigma, k1, k2, data_range, color space, resize
  policy, and domain tag).

Reference implementations and documentation:
* Wang et al., "Image Quality Assessment: From Error Visibility to Structural
  Similarity" (2004).
* torchmetrics.image.StructuralSimilarityIndexMeasure
  (gaussian_kernel=True, kernel_size=11, sigma=1.5, k1=0.01, k2=0.03).
* scikit-image.metrics structural_similarity (emphasises correct data_range).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure as TMSSIM


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _ensure_batch_dim(tensor: Tensor) -> Tensor:
    """Ensure the tensor has a batch dimension (accepts CxHxW or NxCxHxW)."""

    if tensor.ndim == 4:
        return tensor
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    raise ValueError(
        "SSIM expects tensors with 3 (C,H,W) or 4 (N,C,H,W) dimensions. "
        f"Received shape {tuple(tensor.shape)}."
    )


def _auto_data_range(x: Tensor, y: Tensor) -> float:
    """Infer a common data range among [0,1], [-1,1], or [0,255]."""

    xmin = min(float(x.min().item()), float(y.min().item()))
    xmax = max(float(x.max().item()), float(y.max().item()))
    if xmax <= 1.0 and xmin >= 0.0:
        return 1.0
    if xmax <= 1.0 and xmin >= -1.0:
        return 2.0
    return 255.0


def _valid_kernel_size(height: int, width: int, kernel_size: int) -> int:
    """Clip kernel_size to an odd integer not exceeding the smallest dimension."""

    k = int(kernel_size)
    if k <= 0:
        raise ValueError(f"kernel_size must be positive, received {kernel_size}.")
    if k % 2 == 0:
        k -= 1
    k = max(1, min(k, height, width))
    if k % 2 == 0:
        k -= 1
    if k < 1:
        raise ValueError(
            f"kernel_size cannot be adjusted to a valid value for shape {(height, width)}."
        )
    return k


def _percentile_95(values: Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    if values.numel() == 1:
        return float(values.item())
    try:
        return float(torch.quantile(values, 0.95).item())
    except (AttributeError, RuntimeError):
        k = max(1, int(math.ceil(0.95 * values.numel())))
        return float(values.kthvalue(k)[0].item())


def _extract_kernel_size(value: Any) -> int:
    """Best-effort extraction of scalar kernel size from torchmetrics storage."""

    if isinstance(value, (list, tuple)) and value and len(value) > 0:
        if isinstance(value[0], (int, float, str)):
            return int(value[0])
        raise ValueError(f"Cannot convert kernel size from sequence: {value}")
    if isinstance(value, (int, float, str)):
        return int(value)
    raise ValueError(f"Cannot extract kernel size from value: {value}")


def _to_luma_bt601(images: Tensor) -> Tensor:
    """Convert RGB images (N,3,H,W) to BT.601 luma channel (N,1,H,W)."""

    if images.shape[1] != 3:
        raise ValueError(
            f"color_space='y' expects 3-channel RGB input, got C={images.shape[1]}."
        )
    r = images[:, 0:1]
    g = images[:, 1:2]
    b = images[:, 2:3]
    y_channel = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return y_channel


def _align_pair(
    target: Tensor,
    prediction: Tensor,
    policy: Optional[Literal["resize", "center_crop"]],
    mode: Literal["bilinear", "bicubic"] = "bilinear",
) -> Tuple[Tensor, Tensor]:
    """Align spatial dimensions for a pair of tensors according to policy."""

    if policy is None:
        if target.shape[-2:] != prediction.shape[-2:]:
            raise ValueError(
                "SSIM requires equal spatial dimensions when no resize_policy is set. "
                f"Got target={target.shape[-2:]}, prediction={prediction.shape[-2:]}"
            )
        return target, prediction

    if policy == "resize":
        prediction = F.interpolate(
            prediction,
            size=target.shape[-2:],
            mode=mode,
            align_corners=False if mode in {"bilinear", "bicubic"} else None,
        )
        return target, prediction

    if policy == "center_crop":
        h = min(target.shape[-2], prediction.shape[-2])
        w = min(target.shape[-1], prediction.shape[-1])

        def _crop(x: Tensor, hh: int, ww: int) -> Tensor:
            _, _, H, W = x.shape
            top = max((H - hh) // 2, 0)
            left = max((W - ww) // 2, 0)
            return x[:, :, top : top + hh, left : left + ww]

        return _crop(target, h, w), _crop(prediction, h, w)

    raise ValueError(f"Unknown resize_policy '{policy}'. Use None, 'resize', or 'center_crop'.")


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------


@dataclass
class _SSIMConfig:
    kernel_size: int = 11
    sigma: float = 1.5
    k1: float = 0.01
    k2: float = 0.03
    color_space: Literal["rgb", "y"] = "rgb"
    resize_policy: Optional[Literal["resize", "center_crop"]] = None
    resize_mode: Literal["bilinear", "bicubic"] = "bilinear"
    reduction: Literal["none", "elementwise_mean"] = "none"
    domain: Optional[Literal["linear", "srgb"]] = None


class SSIMEvaluator:
    """Model-agnostic SSIM evaluator returning per-image scores and summaries."""

    def __init__(
        self,
        *,
        kernel_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        data_range: Optional[float] = None,
        color_space: Literal["rgb", "y"] = "rgb",
        resize_policy: Optional[Literal["resize", "center_crop"]] = None,
        resize_mode: Literal["bilinear", "bicubic"] = "bilinear",
        reduction: Literal["none", "elementwise_mean"] = "none",
        domain: Optional[Literal["linear", "srgb"]] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.cfg = _SSIMConfig(
            kernel_size=kernel_size,
            sigma=sigma,
            k1=k1,
            k2=k2,
            color_space=color_space,
            resize_policy=resize_policy,
            resize_mode=resize_mode,
            reduction=reduction,
            domain=domain,
        )
        self._metric: Optional[TMSSIM] = None
        self._explicit_data_range = data_range
        self._device = torch.device(device) if device is not None else None

    def _device_or_default(self) -> torch.device:
        if self._device is not None:
            return self._device
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_metric(self, height: int, width: int, data_range: float) -> TMSSIM:
        kernel = _valid_kernel_size(height, width, self.cfg.kernel_size)
        self.cfg.kernel_size = kernel

        rebuild = False
        if self._metric is None:
            rebuild = True
        else:
            current_kernel = getattr(self._metric, "kernel_size", None)
            current_range = getattr(self._metric, "data_range", None)
            if (
                current_kernel is None
                or _extract_kernel_size(current_kernel) != kernel
                or current_range != data_range
            ):
                rebuild = True

        # Always rebuild to ensure consistent configuration
        metric = TMSSIM(
            gaussian_kernel=True,
            sigma=self.cfg.sigma,
            kernel_size=kernel,
            data_range=data_range,
            k1=self.cfg.k1,
            k2=self.cfg.k2,
            reduction=self.cfg.reduction,
        )
        self._metric = metric.to(self._device_or_default()).eval()
        return self._metric

    def _prepare_inputs(self, target: Tensor, prediction: Tensor) -> Tuple[Tensor, Tensor]:
        target = _ensure_batch_dim(target)
        prediction = _ensure_batch_dim(prediction)
        if target.shape != prediction.shape:
            # Allow spatial mismatch to be handled downstream by resize policy.
            if target.shape[0] != prediction.shape[0] or target.shape[1] != prediction.shape[1]:
                raise ValueError(
                    "SSIM requires the same batch size and channel count for target and prediction. "
                    f"Got target={target.shape}, prediction={prediction.shape}."
                )

        device = self._device_or_default()
        target = target.to(device=device, dtype=torch.float32)
        prediction = prediction.to(device=device, dtype=torch.float32)

        target, prediction = _align_pair(target, prediction, self.cfg.resize_policy, self.cfg.resize_mode)

        if self.cfg.color_space == "y":
            if target.shape[1] == 3:
                target = _to_luma_bt601(target)
                prediction = _to_luma_bt601(prediction)
            elif target.shape[1] == 1:
                # Already single-channel; nothing to convert.
                pass
            else:
                raise ValueError(
                    f"color_space='y' expects inputs with 1 or 3 channels, got C={target.shape[1]}."
                )
        elif self.cfg.color_space != "rgb":
            raise ValueError(f"Unsupported color_space '{self.cfg.color_space}'. Use 'rgb' or 'y'.")

        return target, prediction

    @torch.no_grad()
    def __call__(self, ground_truth: Tensor, prediction: Tensor) -> Dict[str, Any]:
        """Compute SSIM scores and summary statistics for a pair or batch."""

        target, pred = self._prepare_inputs(ground_truth, prediction)
        data_range = (
            float(self._explicit_data_range)
            if self._explicit_data_range is not None
            else _auto_data_range(target, pred)
        )

        metric = self._build_metric(target.shape[-2], target.shape[-1], data_range)

        scores = metric(pred, target)
        scores = scores.view(-1)

        # Summary statistics
        mean = float(scores.mean().item())
        std = float(scores.std(unbiased=False).item()) if scores.numel() > 1 else 0.0
        median = float(scores.median().item())
        p95 = _percentile_95(scores)

        result: Dict[str, Any] = {
            "per_image": [float(v) for v in scores.cpu().tolist()],
            "mean": mean,
            "std": std,
            "p50": median,
            "p95": p95,
            "count": int(scores.numel()),
            "kernel_size": _extract_kernel_size(metric.kernel_size),
            "sigma": float(self.cfg.sigma),
            "k1": float(self.cfg.k1),
            "k2": float(self.cfg.k2),
            "data_range": float(data_range),
            "color_space": self.cfg.color_space,
            "resize_policy": self.cfg.resize_policy,
            "domain": self.cfg.domain,
        }
        return result


# Alias retained for readability in other modules if needed.
SSIMMetricUnified = SSIMEvaluator


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def calculate_ssim(
    img_true: Tensor,
    img_pred: Tensor,
    data_range: float,
    *,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    win_size: Optional[int] = None,
    color_space: Literal["rgb", "y"] = "rgb",
    resize_policy: Optional[Literal["resize", "center_crop"]] = None,
    resize_mode: Literal["bilinear", "bicubic"] = "bilinear",
    domain: Optional[Literal["linear", "srgb"]] = None,
) -> float:
    """Compute the average SSIM score between two tensors."""

    if data_range <= 0:
        raise ValueError(f"data_range must be positive, received {data_range}.")

    if win_size is not None:
        kernel_size = int(win_size)

    evaluator = SSIMEvaluator(
        kernel_size=kernel_size,
        sigma=sigma,
        k1=k1,
        k2=k2,
        data_range=data_range,
        color_space=color_space,
        resize_policy=resize_policy,
        resize_mode=resize_mode,
        domain=domain,
        reduction="elementwise_mean",
    )
    stats = evaluator(img_true, img_pred)
    return stats["mean"]


@torch.no_grad()
def calculate_ssim_per_image(
    img_true: Tensor,
    img_pred: Tensor,
    data_range: float,
    *,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    win_size: Optional[int] = None,
    color_space: Literal["rgb", "y"] = "rgb",
    resize_policy: Optional[Literal["resize", "center_crop"]] = None,
    resize_mode: Literal["bilinear", "bicubic"] = "bilinear",
    domain: Optional[Literal["linear", "srgb"]] = None,
) -> Tensor:
    """Return per-image SSIM scores for a batch."""

    if data_range <= 0:
        raise ValueError(f"data_range must be positive, received {data_range}.")

    if win_size is not None:
        kernel_size = int(win_size)

    evaluator = SSIMEvaluator(
        kernel_size=kernel_size,
        sigma=sigma,
        k1=k1,
        k2=k2,
        data_range=data_range,
        color_space=color_space,
        resize_policy=resize_policy,
        resize_mode=resize_mode,
        domain=domain,
        reduction="none",
    )
    stats = evaluator(img_true, img_pred)
    return torch.tensor(stats["per_image"], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Batch helpers and legacy streaming interface
# ---------------------------------------------------------------------------


def evaluate_pairs_ssim(
    pairs: List[Tuple[Tensor, Tensor]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate multiple (ground_truth, prediction) pairs under one configuration."""

    evaluator = SSIMEvaluator(**kwargs)
    scores: List[float] = []
    last_stats: Optional[Dict[str, Any]] = None
    for gt, pred in pairs:
        stats = evaluator(gt, pred)
        scores.extend(stats["per_image"])
        last_stats = stats

    if not scores:
        return {
            "per_image": [],
            "mean": float("nan"),
            "std": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "count": 0,
        }

    tensor_scores = torch.tensor(scores, dtype=torch.float32)
    mean = float(tensor_scores.mean().item())
    std = float(tensor_scores.std(unbiased=False).item()) if tensor_scores.numel() > 1 else 0.0
    median = float(tensor_scores.median().item())
    p95 = _percentile_95(tensor_scores)

    summary: Dict[str, Any] = {
        "per_image": [float(v) for v in tensor_scores.tolist()],
        "mean": mean,
        "std": std,
        "p50": median,
        "p95": p95,
        "count": int(tensor_scores.numel()),
    }
    if last_stats is not None:
        summary.update(
            {
                "kernel_size": last_stats["kernel_size"],
                "sigma": last_stats["sigma"],
                "k1": last_stats["k1"],
                "k2": last_stats["k2"],
                "color_space": last_stats["color_space"],
                "resize_policy": last_stats["resize_policy"],
                "data_range": last_stats["data_range"],
                "domain": last_stats["domain"],
            }
        )
    return summary


class SSIMMetric:
    """Streaming SSIM aggregator (backward-compatible helper).

    Maintains mean/std/count across repeated calls to ``update`` while relying on
    the evaluator implementation to ensure consistent configuration.
    """

    def __init__(
        self,
        *,
        data_range: float,
        kernel_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        color_space: Literal["rgb", "y"] = "rgb",
        device: Optional[torch.device | str] = None,
        win_size: Optional[int] = None,
        resize_policy: Optional[Literal["resize", "center_crop"]] = None,
        resize_mode: Literal["bilinear", "bicubic"] = "bilinear",
        domain: Optional[Literal["linear", "srgb"]] = None,
    ) -> None:
        if data_range <= 0:
            raise ValueError("data_range must be positive.")
        if win_size is not None:
            kernel_size = int(win_size)
        self._evaluator = SSIMEvaluator(
            kernel_size=kernel_size,
            sigma=sigma,
            k1=k1,
            k2=k2,
            data_range=data_range,
            color_space=color_space,
            resize_policy=resize_policy,
            resize_mode=resize_mode,
            domain=domain,
            reduction="none",
            device=device,
        )
        self.reset()

    def reset(self) -> None:
        self._sum = 0.0
        self._sumsq = 0.0
        self._count = 0

    @torch.no_grad()
    def update(self, img_true: Tensor, img_pred: Tensor) -> None:
        stats = self._evaluator(img_true, img_pred)
        scores = torch.tensor(stats["per_image"], dtype=torch.float64)
        self._sum += float(scores.sum().item())
        self._sumsq += float((scores ** 2).sum().item())
        self._count += int(scores.numel())

    def compute(self) -> Dict[str, float]:
        if self._count == 0:
            return {"mean": float("nan"), "std": float("nan"), "count": 0}
        mean = self._sum / self._count
        var = max(self._sumsq / self._count - mean * mean, 0.0)
        return {"mean": mean, "std": var ** 0.5, "count": int(self._count)}


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    torch.manual_seed(0)

    # Example: random 8-bit RGB data with automatic range detection.
    batch_size, channels, height, width = 4, 3, 128, 128
    reference = torch.randint(0, 256, (batch_size, channels, height, width), dtype=torch.uint8)
    prediction = reference.float() + torch.randn(batch_size, channels, height, width) * 5.0
    prediction = prediction.clamp(0, 255).to(dtype=torch.uint8)

    reference = reference.float()
    prediction = prediction.float()

    evaluator = SSIMEvaluator(kernel_size=11, sigma=1.5, resize_policy=None)
    summary = evaluator(reference, prediction)
    print(
        "[SSIM] mean={mean:.6f}, std={std:.6f}, p50={p50:.6f}, p95={p95:.6f}, "
        "kernel={kernel_size}, sigma={sigma}, data_range={data_range}"
        .format(**summary)
    )

    # Streaming usage stays compatible with previous interface.
    stream = SSIMMetric(data_range=255.0)
    stream.update(reference, prediction)
    print("[SSIM-Streaming]", stream.compute())

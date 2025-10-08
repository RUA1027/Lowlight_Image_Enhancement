# Dependency: torchmetrics (pip install torchmetrics)
"""
SSIM (Structural Similarity Index Measure) computation utility.

This module provides a thin wrapper around ``torchmetrics.StructuralSimilarityIndexMeasure``
so that heterogeneous restoration models (NewBP-NAFNet, vanilla NAFNet, U-Net, SwinIR, etc.)
can share the same evaluation logic when executed in environments like Google Colab.
"""

from __future__ import annotations

from typing import Dict

import torch
from torchmetrics import StructuralSimilarityIndexMeasure


def _ensure_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure the tensor has an explicit batch dimension."""
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    raise ValueError(
        f"Input tensor must have 3 (C,H,W) or 4 (N,C,H,W) dimensions, received shape {tuple(tensor.shape)}."
    )


def calculate_ssim(
    img_true: torch.Tensor,
    img_pred: torch.Tensor,
    data_range: float,
    win_size: int = 7,
) -> float:
    """Compute the Structural Similarity Index Measure (SSIM) between two images.

    This function is a convenience wrapper around
    :class:`torchmetrics.StructuralSimilarityIndexMeasure`, automatically handling batch
    dimensions and device placement so that it can be reused across multiple restoration
    models evaluated under the same metric suite.

    Args:
        img_true: Ground-truth image tensor of shape ``(C, H, W)`` or ``(N, C, H, W)``.
        img_pred: Predicted image tensor with the same shape as ``img_true``.
        data_range: Maximum possible intensity value (e.g., ``1.0`` for normalized inputs
            or ``255.0`` for 8-bit images).
        win_size: Odd integer specifying the size of the sliding window used in SSIM
            computation. Defaults to ``7``.

    Returns:
        Average SSIM score as a Python ``float``. Values typically lie within ``[-1, 1]``.

    Raises:
        ValueError: If the input tensors do not share the same shape, have unsupported
            dimensionality, if ``data_range`` is non-positive, or if ``win_size`` is not
            a positive odd integer.

    Example:
        >>> import torch
        >>> from metrics.ssim import calculate_ssim
        >>> true_image = torch.rand(1, 32, 32)
        >>> pred_image = torch.clamp(true_image + 0.05 * torch.randn_like(true_image), 0, 1)
        >>> score = calculate_ssim(true_image, pred_image, data_range=1.0)
        >>> isinstance(score, float)
        True

    """
    if img_true.shape != img_pred.shape:
        raise ValueError(
            f"Input shapes must match exactly, got {img_true.shape=} and {img_pred.shape=}."
        )

    if data_range <= 0:
        raise ValueError(f"`data_range` must be positive, received {data_range}.")

    if win_size <= 0 or win_size % 2 == 0:
        raise ValueError(f"`win_size` must be a positive odd integer, received {win_size}.")

    img_true_batched = _ensure_batch_dim(img_true)
    img_pred_batched = _ensure_batch_dim(img_pred)

    ssim_calculator = StructuralSimilarityIndexMeasure(
        data_range=data_range,
        win_size=win_size,
    ).to(img_true_batched.device)

    ssim_value = ssim_calculator(img_pred_batched, img_true_batched)
    return float(ssim_value.item())


@torch.no_grad()
def calculate_ssim_per_image(
    img_true: torch.Tensor,
    img_pred: torch.Tensor,
    data_range: float,
    win_size: int = 7,
) -> torch.Tensor:
    """Compute SSIM for each image in a batch and return a 1D tensor of scores."""
    if img_true.shape != img_pred.shape:
        raise ValueError(
            f"Input shapes must match exactly, got {img_true.shape=} and {img_pred.shape=}."
        )
    if data_range <= 0:
        raise ValueError(f"`data_range` must be positive, received {data_range}.")
    if win_size <= 0 or win_size % 2 == 0:
        raise ValueError(f"`win_size` must be a positive odd integer, received {win_size}.")

    img_true_batched = _ensure_batch_dim(img_true)
    img_pred_batched = _ensure_batch_dim(img_pred)
    device = img_true_batched.device
    metric = StructuralSimilarityIndexMeasure(data_range=data_range, win_size=win_size).to(device)

    scores = []
    for idx in range(img_true_batched.shape[0]):
        score = metric(img_pred_batched[idx : idx + 1], img_true_batched[idx : idx + 1])
        scores.append(score.detach())
        metric.reset()
    return torch.stack(scores, dim=0).flatten()



class SSIMMetric:
    """Streaming SSIM aggregator providing dataset-level statistics."""

    def __init__(self, data_range: float, win_size: int = 7) -> None:
        if data_range <= 0:
            raise ValueError("`data_range` must be positive.")
        if win_size <= 0 or win_size % 2 == 0:
            raise ValueError("`win_size` must be a positive odd integer.")
        self.data_range = data_range
        self.win_size = win_size
        self.reset()

    def reset(self) -> None:
        self._sum = 0.0
        self._sumsq = 0.0
        self._count = 0

    @torch.no_grad()
    def update(self, img_true: torch.Tensor, img_pred: torch.Tensor) -> None:
        vals = calculate_ssim_per_image(
            img_true=img_true,
            img_pred=img_pred,
            data_range=self.data_range,
            win_size=self.win_size,
        )
        self._sum += float(vals.sum().item())
        self._sumsq += float((vals**2).sum().item())
        self._count += int(vals.numel())

    def compute(self) -> Dict[str, float]:
        if self._count == 0:
            mean = float("nan")
            std = float("nan")
        else:
            mean = self._sum / self._count
            var = max(self._sumsq / self._count - mean * mean, 0.0)
            std = var**0.5
        return {"mean": mean, "std": std, "count": float(self._count)}


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    true_image = torch.rand(1, 32, 32)
    pred_image_noise = torch.clamp(true_image + 0.1 * torch.randn(1, 32, 32), 0.0, 1.0)

    ssim_single = calculate_ssim(true_image, pred_image_noise, data_range=1.0)
    print(f"SSIM (single image input): {ssim_single:.4f}")

    ssim_identical = calculate_ssim(true_image, true_image, data_range=1.0)
    print(f"SSIM (identical images): {ssim_identical:.4f}")

    true_batch = torch.rand(4, 1, 32, 32)
    pred_batch = true_batch.clone()
    pred_batch[0] = torch.clamp(pred_batch[0] + 0.5 * torch.randn(1, 32, 32), 0.0, 1.0)

    ssim_batch = calculate_ssim(true_batch, pred_batch, data_range=1.0)
    print(f"SSIM (batch input, averaged): {ssim_batch:.4f}")

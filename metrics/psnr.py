"""
PSNR (Peak Signal-to-Noise Ratio) computation utility.

This module exposes a single function, :func:`calculate_psnr`, which operates purely on
PyTorch tensors. It is designed to be reusable across multiple restoration models
imported from different repositories (e.g., NewBP-NAFNet, vanilla NAFNet, U-Net,
SwinIR) so that the same evaluation logic can be shared when running experiments in
environments such as Google Colab.
"""

from __future__ import annotations

from typing import Union

import torch


def calculate_psnr(img_true: torch.Tensor, img_pred: torch.Tensor, data_range: float) -> float:
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    The function accepts arbitrary batched or unbatched tensors as long as the shapes
    of ``img_true`` and ``img_pred`` match exactly. Internally, computations are cast to
    ``torch.float64`` for numerical stability in mixed-precision settings.

    Args:
        img_true: Ground-truth image tensor. Shape must match ``img_pred`` exactly.
        img_pred: Predicted image tensor (e.g., output from any restoration model).
        data_range: Maximum possible data value (e.g., ``1.0`` for images normalized to
            ``[0, 1]`` or ``255.0`` for 8-bit images).

    Returns:
        PSNR value measured in decibels (dB). Returns ``float('inf')`` when both inputs
        are identical yielding zero mean squared error.

    Raises:
        ValueError: If the two input tensors do not share the same shape or if
            ``data_range`` is non-positive.

    Example:
        >>> import torch
        >>> from metrics.psnr import calculate_psnr
        >>> true_image = torch.rand(3, 256, 256)
        >>> pred_image = true_image + 0.01 * torch.randn_like(true_image)
        >>> pred_image = torch.clamp(pred_image, 0, 1)
        >>> psnr_val = calculate_psnr(true_image, pred_image, data_range=1.0)
        >>> isinstance(psnr_val, float)
        True

    """
    if img_true.shape != img_pred.shape:
        raise ValueError(
            f"Input shapes must match exactly, got {img_true.shape=} and {img_pred.shape=}."
        )

    if data_range <= 0:
        raise ValueError(f"`data_range` must be positive, received {data_range}.")

    img_true64 = img_true.to(dtype=torch.float64)
    img_pred64 = img_pred.to(dtype=torch.float64)

    mse = torch.mean((img_true64 - img_pred64) ** 2)

    if torch.isclose(mse, mse.new_tensor(0.0), atol=1e-12):
        return float("inf")

    psnr = 10.0 * torch.log10((data_range ** 2) / mse)
    return float(psnr.item())


if __name__ == "__main__":
    torch.manual_seed(0)

    true_image = torch.rand(3, 256, 256)
    pred_image_slight_noise = true_image + 0.01 * torch.randn(3, 256, 256)
    pred_image_slight_noise = torch.clamp(pred_image_slight_noise, 0.0, 1.0)

    psnr_val = calculate_psnr(true_image, pred_image_slight_noise, data_range=1.0)
    print(f"PSNR (slight noise): {psnr_val:.4f} dB")

    psnr_inf = calculate_psnr(true_image, true_image, data_range=1.0)
    print(f"PSNR (identical images): {psnr_inf} dB")

    pred_image_high_noise = torch.rand(3, 256, 256)
    psnr_low = calculate_psnr(true_image, pred_image_high_noise, data_range=1.0)
    print(f"PSNR (high noise): {psnr_low:.4f} dB")

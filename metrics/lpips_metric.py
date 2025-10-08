# Dependency: lpips (pip install lpips)
"""
LPIPS (Learned Perceptual Image Patch Similarity) evaluation utility.

This module exposes :class:`LPIPSEvaluator`, a reusable wrapper that loads the LPIPS
network once and can be invoked repeatedly across different restoration models
evaluated under the same metric suite (e.g., NewBP-NAFNet, vanilla NAFNet, U-Net,
SwinIR) in environments such as Google Colab.
"""

from __future__ import annotations

from typing import Literal

import torch
import lpips


class LPIPSEvaluator:
    """Callable helper for computing LPIPS scores over batched image tensors.

    The class encapsulates the underlying LPIPS network so that the model weights are
    loaded a single time during initialization and reused for subsequent evaluations.
    """

    def __init__(self, net: Literal["alex", "vgg", "squeeze"] = "alex", device: str = "cuda") -> None:
        """Initialize the LPIPS evaluator.

        Args:
            net: Backbone network used inside LPIPS. Common choices include ``"alex"``
                (default), ``"vgg"``, and ``"squeeze"``.
            device: Device identifier on which the LPIPS module should reside (e.g.,
                ``"cuda"`` or ``"cpu"``).
        """
        self.device = torch.device(device)
        self.loss_fn = lpips.LPIPS(net=net).to(self.device)
        self.loss_fn.eval()

    def __call__(self, img_true: torch.Tensor, img_pred: torch.Tensor) -> float:
        """Compute the average LPIPS score between two image batches.

        The inputs are expected to be in a standard image range (e.g., ``[0, 1]`` or
        ``[0, 255]``). This method will internally normalize them to ``[-1, 1]`` as
        required by the LPIPS implementation before evaluation.

        Args:
            img_true: Ground-truth image tensor with shape ``(N, C, H, W)`` or ``(C, H, W)``.
            img_pred: Predicted image tensor with the same shape as ``img_true``.

        Returns:
            Average LPIPS distance as a floating-point value (smaller values indicate
            higher perceptual similarity).

        Raises:
            ValueError: If the input tensors do not have matching shapes or if their
                dimensionality is unsupported.
        """
        if img_true.shape != img_pred.shape:
            raise ValueError(
                f"Input shapes must match exactly, got {img_true.shape=} and {img_pred.shape=}."
            )

        if img_true.ndim not in (3, 4):
            raise ValueError(
                f"Inputs must be 3D (C,H,W) or 4D (N,C,H,W) tensors, received ndim={img_true.ndim}."
            )

        if img_true.ndim == 3:
            img_true = img_true.unsqueeze(0)
            img_pred = img_pred.unsqueeze(0)

        img_true = img_true.to(self.device, dtype=torch.float32)
        img_pred = img_pred.to(self.device, dtype=torch.float32)

        img_true_norm = img_true * 2.0 - 1.0
        img_pred_norm = img_pred * 2.0 - 1.0

        with torch.no_grad():
            distances = self.loss_fn(img_pred_norm, img_true_norm)

        return float(distances.mean().item())


if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = LPIPSEvaluator(net="alex", device=device)
    print(f"LPIPS evaluator initialized on {device.upper()}.")

    true_images_batch = torch.rand(4, 3, 64, 64)
    pred_images_batch_noisy = torch.clamp(true_images_batch + 0.1 * torch.randn(4, 3, 64, 64), 0.0, 1.0)

    lpips_score = evaluator(true_images_batch, pred_images_batch_noisy)
    print(f"LPIPS score (noisy images): {lpips_score:.4f}")

    lpips_score_identical = evaluator(true_images_batch, true_images_batch)
    print(f"LPIPS score (identical images): {lpips_score_identical:.4f}")

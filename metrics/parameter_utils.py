"""Utilities for analyzing model parameter counts across different architectures."""

# Dependency: torch (pip install torch)

from __future__ import annotations

from typing import Literal

import torch


def count_parameters(model: torch.nn.Module, unit: Literal["raw", "K", "M"] = "M") -> float:
    """Count trainable parameters in a PyTorch module.

    Args:
        model: Model whose trainable parameter count is queried.
        unit: Unit for the returned value. ``"raw"`` returns the exact count,
            ``"K"`` scales by 1e3, and ``"M"`` (default) scales by 1e6.

    Returns:
        Number of trainable parameters represented as a float in the requested unit.

    Raises:
        ValueError: If ``unit`` is not one of ``{"raw", "K", "M"}``.
    """

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if unit == "raw":
        return float(total_params)
    if unit == "K":
        return float(total_params) / 1e3
    if unit == "M":
        return float(total_params) / 1e6
    raise ValueError("`unit` must be one of {'raw', 'K', 'M'}.")


if __name__ == "__main__":
    import torch.nn as nn

    class SimpleCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.register_buffer("non_trainable_buffer", torch.randn(10))
            self.frozen_param = nn.Parameter(torch.randn(5), requires_grad=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - demo only
            x = self.relu(self.conv1(x))
            return self.relu(self.conv2(x))

    model = SimpleCNN()
    print("--- Analyzing SimpleCNN model ---")

    params_m = count_parameters(model, unit="M")
    print(f"Trainable parameters: {params_m:.4f} M")

    params_k = count_parameters(model, unit="K")
    print(f"Trainable parameters: {params_k:.2f} K")

    params_raw = count_parameters(model, unit="raw")
    print(f"Trainable parameters: {params_raw} (raw count)")

    manual = 432 + 16 + 4608 + 32
    print(f"Manual calculation for verification: {manual}")
    assert params_raw == manual
    print("Verification successful: function only counts trainable parameters.")

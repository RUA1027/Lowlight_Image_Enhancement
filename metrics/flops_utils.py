# Dependency: fvcore (pip install 'fvcore @ git+https://github.com/facebookresearch/fvcore')
"""Utilities for computing model FLOPs across diverse architectures."""

from __future__ import annotations

from typing import Literal

import torch
from fvcore.nn import FlopCountAnalysis


def _infer_model_device(model: torch.nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def count_flops(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    unit: Literal["M", "G"] = "G",
) -> float:
    """Count floating point operations for a single forward pass.

    Args:
        model: PyTorch module to analyse (evaluated in inference mode).
        input_tensor: Example input tensor whose shape/dtype/device match real usage.
        unit: Reporting unit. ``"M"`` for mega-FLOPs, ``"G"`` (default) for giga-FLOPs.

    Returns:
        FLOP count as a floating-point number in the requested unit.

    Raises:
        ValueError: If ``unit`` is not ``"M"`` or ``"G"`` or if input/device mismatch.
    """

    if unit not in {"M", "G"}:
        raise ValueError("`unit` must be one of {'M', 'G'}.")

    model_device = _infer_model_device(model)
    if input_tensor.device != model_device:
        raise ValueError("Model and input tensor must reside on the same device.")

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            analysis = FlopCountAnalysis(model, input_tensor)
            flops = analysis.total()
    finally:
        if was_training:
            model.train()

    scale = 1e6 if unit == "M" else 1e9
    return float(flops) / scale


if __name__ == "__main__":
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3)
            self.fc = nn.Linear(16 * 30 * 30, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - demo only
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleModel().to(device)
    dummy_input = torch.randn(1, 3, 32, 32, device=device)

    gflops = count_flops(model, dummy_input, unit="G")
    print(f"Model GFLOPs for a (1, 3, 32, 32) input: {gflops:.4f} GFLOPs")

    mflops = count_flops(model, dummy_input, unit="M")
    print(f"Model MFLOPs for a (1, 3, 32, 32) input: {mflops:.2f} MFLOPs")

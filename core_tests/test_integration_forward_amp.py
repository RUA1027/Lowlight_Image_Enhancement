"""
Integration goals:
- Run end-to-end AMP (autocast + GradScaler) training for mono/rgb PSF modes and confirm the loop completes without NaN/Inf.
- Verify loss decreases over several steps, optimizer only updates the backbone, and PSF kernels remain fixed buffers.
- Ensure CUDA availability gating, deterministic seed setup, and AMP best practices are respected.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# 将项目根目录和 NAFNet 目录添加到 Python 路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
nafnet_root = os.path.join(project_root, 'NAFNet')
for path in [project_root, nafnet_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from NewBP_model.newbp_net_arch import create_crosstalk_psf, create_newbp_net
    from NewBP_model.losses import PhysicalConsistencyLossSRGB
except ModuleNotFoundError as exc:  # pragma: no cover
    pytest.skip(f"Skipping AMP integration tests due to missing dependency: {exc}", allow_module_level=True)


def _fix_seed(seed: int = 2025) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _toy_batch(
    *,
    batch: int = 4,
    channels: int = 3,
    height: int = 96,
    width: int = 96,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.rand(batch, channels, height, width, device=device)
    b = torch.rand(batch, channels, height, width, device=device)
    ratio = torch.ones(batch, device=device)
    return a, b, ratio


def _make_psf(mode: str, device: torch.device) -> nn.Module:
    spec = "P2" if mode == "mono" else "B2"
    return create_crosstalk_psf(psf_mode=mode, kernel_spec=spec).to(device)


class ScenarioBWrapper(nn.Module):
    """
    Lightweight Scenario-B wrapper combining a NAFNet backbone with a Crosstalk PSF buffer.
    """

    def __init__(self, mode: str, device: torch.device) -> None:
        super().__init__()
        self.backbone = create_newbp_net(in_channels=3)
        self.psf = _make_psf(mode, torch.device("cpu"))
        self._mode = mode
        self.to(device)

    @property
    def mode(self) -> str:
        return self._mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _opt_param_count(params: Iterable[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP tests")
@pytest.mark.parametrize("mode", ["mono", "rgb"])
def test_amp_end_to_end_training_smoke(mode: str) -> None:
    """
    AMP smoke test:
    - Train Scenario-B wrapper with autocast + GradScaler for several iterations.
    - Loss should decrease, optimizer updates only backbone, PSF buffer remains fixed.
    """
    _fix_seed(2025)
    device = torch.device("cuda")

    model = ScenarioBWrapper(mode=mode, device=device)
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=2e-3)

    backbone_param_count = _opt_param_count(model.backbone.parameters())
    optimizer_param_count = sum(_opt_param_count(group["params"]) for group in optimizer.param_groups)
    assert backbone_param_count == optimizer_param_count, "Optimizer should only manage backbone parameters."

    scaler = torch.cuda.amp.GradScaler()
    phys_loss = PhysicalConsistencyLossSRGB(psf_module=model.psf)

    losses: list[float] = []
    first_backbone_name, first_backbone_before = next(iter(model.backbone.state_dict().items()))
    first_backbone_before = first_backbone_before.detach().clone()
    psf_kernel_before = model.psf.kernel.detach().clone()

    for _ in range(5):
        a, b, ratio = _toy_batch(device=device)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            bhat = model(a)
            pix = F.l1_loss(bhat, b)
            phys = phys_loss(bhat, a, ratio)
            loss = pix + phys

        assert torch.isfinite(loss), "Loss produced NaN/Inf under AMP."

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.detach().item()))

    assert losses[-1] < losses[0] * 0.99, f"Loss did not show a downward trend under AMP: {losses!r}"

    first_backbone_after = model.backbone.state_dict()[first_backbone_name]
    assert not torch.allclose(first_backbone_after, first_backbone_before), "Backbone weights failed to update."
    assert torch.allclose(model.psf.kernel, psf_kernel_before), "PSF kernel buffer should remain unchanged."

    assert all(torch.isfinite(torch.tensor(value)) for value in losses), "Loss history contains non-finite values."


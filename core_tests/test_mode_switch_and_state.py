"""
Validation focus:
- Crosstalk PSF kernels live as buffers: they travel with state_dict/to(device) yet never join the optimizer parameter groups.
- Switching mono <-> rgb PSF modes must not mutate the NAFNet backbone weights; forward shape stays intact.
- Saving/loading with strict flags, device transfers, and optimizer steps affect only the backbone while PSF buffers remain untouched.
"""

from __future__ import annotations

import os
import sys

# 将项目根目录和 NAFNet 目录添加到 Python 路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
nafnet_root = os.path.join(project_root, 'NAFNet')
for path in [project_root, nafnet_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

import copy
from typing import Dict, Iterable

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from NewBP_model.newbp_net_arch import create_crosstalk_psf, create_newbp_net
except ModuleNotFoundError as exc:  # pragma: no cover
    pytest.skip(f"Skipping mode/state tests due to missing dependency: {exc}", allow_module_level=True)


def _make_psf(mode: str, device: torch.device) -> nn.Module:
    spec = "P2" if mode == "mono" else "B2"
    return create_crosstalk_psf(psf_mode=mode, kernel_spec=spec).to(device)


class ScenarioBWrapper(nn.Module):
    """
    Minimal Scenario-B style wrapper:
    - `backbone`: standard NAFNet.
    - `psf`: CrosstalkPSF used in loss graph only (buffer-based kernels).
    """

    def __init__(self, mode: str = "mono", device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.backbone = create_newbp_net(in_channels=3)
        self.psf = _make_psf(mode, torch.device("cpu"))
        self._mode = mode
        self.to(device)

    @property
    def mode(self) -> str:
        return self._mode

    def set_psf_mode(self, mode: str) -> None:
        device = next(self.backbone.parameters()).device
        self.psf = _make_psf(mode, device)
        self._mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _rand_img(batch: int = 2, channels: int = 3, height: int = 32, width: int = 32, *, device: str = "cpu") -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(batch, channels, height, width, device=device)


def _tensor_dict_clone(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in sd.items()}


@pytest.mark.parametrize("start_mode,end_mode", [("mono", "rgb"), ("rgb", "mono")])
def test_mode_switch_keeps_backbone_identical(start_mode: str, end_mode: str) -> None:
    model = ScenarioBWrapper(mode=start_mode)
    backbone_before = _tensor_dict_clone(model.backbone.state_dict())

    model.set_psf_mode(end_mode)

    for name, param in model.backbone.state_dict().items():
        assert torch.allclose(param, backbone_before[name]), f"Backbone parameter changed after PSF switch: {name}"


def test_psf_is_buffer_and_in_state_dict() -> None:
    model = ScenarioBWrapper(mode="mono")
    assert sum(p.numel() for p in model.psf.parameters()) == 0, "PSF kernels must be buffers, not Parameters."

    state_keys = list(model.state_dict().keys())
    assert any(key.endswith("psf.kernel") for key in state_keys), "State dict should contain psf.kernel buffer."


def _opt_param_count(params: Iterable[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in params)


def test_optimizer_updates_backbone_only() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ScenarioBWrapper(mode="rgb", device=device)

    optimizer = torch.optim.Adam(model.backbone.parameters(), lr=1e-2)
    backbone_param_count = _opt_param_count(model.backbone.parameters())
    optimizer_param_count = sum(_opt_param_count(group["params"]) for group in optimizer.param_groups)
    assert backbone_param_count == optimizer_param_count

    inputs = _rand_img(device=device)
    targets = torch.zeros_like(inputs)
    output = model(inputs)
    loss = F.mse_loss(output, targets)

    kernel_before = model.psf.kernel.detach().clone()
    first_name, first_param_before = next(iter(model.backbone.state_dict().items()))
    first_param_before = first_param_before.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    first_param_after = model.backbone.state_dict()[first_name]
    assert not torch.allclose(first_param_after, first_param_before), "Backbone parameters did not update."
    assert torch.allclose(model.psf.kernel, kernel_before), "PSF kernel should remain unchanged by optimizer steps."


def test_state_dict_save_load_strictness() -> None:
    model = ScenarioBWrapper(mode="mono")
    sd_original = copy.deepcopy(model.state_dict())

    with torch.no_grad():
        model.psf.kernel.add_(0.5)
    incompatible = model.load_state_dict(sd_original, strict=True)
    assert incompatible.missing_keys == [] and incompatible.unexpected_keys == []

    psf_key = next(key for key in sd_original if key.endswith("psf.kernel"))
    assert torch.allclose(model.state_dict()[psf_key], sd_original[psf_key])

    sd_extra = copy.deepcopy(sd_original)
    sd_extra["psf.extra_dummy"] = torch.tensor(0.0)
    incompatible_relaxed = model.load_state_dict(sd_extra, strict=False)
    assert incompatible_relaxed.missing_keys == []
    assert "psf.extra_dummy" in incompatible_relaxed.unexpected_keys


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_transfer_moves_buffers_and_params() -> None:
    model = ScenarioBWrapper(mode="rgb", device="cpu")

    model = model.to("cuda")
    kernel_device = model.psf.kernel.device
    backbone_device = next(iter(model.backbone.parameters())).device
    assert kernel_device.type == "cuda" and backbone_device.type == "cuda"

    model = model.to("cpu")
    assert model.psf.kernel.device.type == "cpu"
    assert next(iter(model.backbone.parameters())).device.type == "cpu"


@pytest.mark.parametrize("mode", ["mono", "rgb"])
def test_forward_shape_invariant_after_mode_switch(mode: str) -> None:
    model = ScenarioBWrapper(mode=mode)
    inputs = _rand_img(batch=1, height=40, width=48)

    outputs = model(inputs)
    assert outputs.shape == inputs.shape

    switched_mode = "rgb" if mode == "mono" else "mono"
    model.set_psf_mode(switched_mode)
    outputs_switched = model(inputs)
    assert outputs_switched.shape == inputs.shape


# Remarks:
# - PSF kernels are buffers by design: they appear in state_dict and migrate with .to(device) yet never join parameters().
# - strict=True load restores exact saved states, while strict=False tolerates structural experimentation.
# - Optimizers should operate solely on backbone weights; PSF buffers remain fixed guidance for loss computations.

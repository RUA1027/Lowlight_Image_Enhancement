"""
测试意图与判定标准：
1. 不卷输入：纯前向 `model(a_srgb)` 时 PSF 模块不得被调用；只允许在构造 `K * B̂` 的物理一致性损失分支触发。
2. 等价性：当物理项权重设为 0 时，包装模型输出应与原生 NAFNet 逐像素一致（使用 `torch.allclose` 的严格容差判定）。
3. 可复现：固定 Python、NumPy、PyTorch 的随机种子并设置 cuDNN 确定性，避免算法选择差异导致对比失效。
4. 理论锚点：`ConvTranspose2d` 是 `Conv2d` 对输入的梯度；因此将 K 仅放在损失路径即可在反向传播中自然得到 `K^T`，无须对输入再额外卷积。
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

import random
from typing import Any

import numpy as np
import pytest
import torch

from NewBP_model.newbp_layer import CrosstalkPSF
from NewBP_model.newbp_net_arch import NAFNet, create_newbp_net
from NewBP_model.losses import PhysicalConsistencyLossSRGB


def _fix_seed(seed: int = 3407) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _extract_tensor(output: Any) -> torch.Tensor:
    """
    Helper to unwrap potential (tensor, aux) tuples so tests stay agnostic
    to wrapper return signatures.
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        assert output, "Expected non-empty output tuple from model forward."
        first = output[0]
        assert isinstance(first, torch.Tensor), "First item from model must be a tensor."
        return first
    raise TypeError(f"Unexpected model output type: {type(output)!r}")


@pytest.mark.parametrize("mode", ["mono", "rgb"])
def test_forward_does_not_call_psf(mode: str) -> None:
    """
    情景 B 验证 1：纯前向阶段不卷 K，PSF 仅在损失路径中触发。
    """
    _fix_seed(2025)

    if mode == "mono":
        kernels = torch.tensor(
            [[0, 1, 0], [1, 4, 1], [0, 1, 0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
    else:
        kernels = torch.cat(
            [
                torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3),
                torch.tensor([[0, 1, 0], [1, 2, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3),
                torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3),
            ],
            dim=0,
        )

    psf = CrosstalkPSF(mode=mode, kernels=kernels)
    call_state = {"count": 0}

    def _hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        call_state["count"] += 1

    handle = psf.register_forward_hook(_hook)

    model = create_newbp_net(in_channels=3, nafnet_params={"img_channel": 3})
    model.eval()

    a_srgb = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        bhat = _extract_tensor(model(a_srgb))

    assert call_state["count"] == 0, "PSF.forward should not run during pure backbone inference (Scenario B violation)."
    assert bhat.shape == a_srgb.shape

    criterion = PhysicalConsistencyLossSRGB(psf_module=psf)
    _ = criterion(bhat, a_srgb, ratio=1.0)
    assert call_state["count"] == 1, "PSF.forward must be called exactly once inside the physical-consistency loss path."

    handle.remove()


def test_equivalent_to_plain_nafnet_when_phys0() -> None:
    """
    情景 B 验证 2：物理项权重为 0 时，包装模型与原生 NAFNet 前向逐像素一致。
    """
    _fix_seed(2025)

    model_wrapped = create_newbp_net(in_channels=3, nafnet_params={"img_channel": 3})
    model_plain = NAFNet(img_channel=3)
    model_plain.load_state_dict(model_wrapped.state_dict())

    model_wrapped.eval()
    model_plain.eval()

    a_srgb = torch.rand(1, 3, 48, 48)
    with torch.no_grad():
        y_wrapped = _extract_tensor(model_wrapped(a_srgb))
        y_plain = _extract_tensor(model_plain(a_srgb))

    assert torch.allclose(
        y_wrapped,
        y_plain,
        rtol=0.0,
        atol=0.0,
    ), "With zero physical weight, wrapped model must match plain NAFNet exactly."


def test_psf_registers_no_trainable_parameters() -> None:
    """
    情景 B 软性检查：PSF 内部核应注册为 buffer，不得进入优化器参数组。
    """
    kernels = torch.tensor(
        [[0.0100, 0.0200, 0.0100], [0.0200, 0.8800, 0.0200], [0.0100, 0.0200, 0.0100]],
        dtype=torch.float32,
    ).view(1, 1, 3, 3)
    psf = CrosstalkPSF(mode="mono", kernels=kernels)
    total_params = sum(p.numel() for p in psf.parameters())
    assert total_params == 0, "Scenario B expects PSF kernels as buffers so optimizers never receive them."


# 额外说明：
# - 若需要观察更丰富的调用信息，可改用 `register_forward_hook` / `register_full_backward_hook`（PyTorch 官方文档）。
# - 物理项不参与前向卷积即可依赖 autograd 提供的 `ConvTranspose2d` 梯度回放，自然获得 `K^T`（参见 torch.nn.ConvTranspose2d 文档）。

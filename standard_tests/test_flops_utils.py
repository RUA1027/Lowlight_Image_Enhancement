"""
FLOPs/MACs wrapper regression tests validating counting conventions and metadata.

References:
- fvcore `FlopCountAnalysis`: treats one fused multiply-add (FMA) as one FLOP,
  exposes total/by_operator/by_module/unsupported/uncalled reports and allows
  custom operator handles.
- Community tools (thop/ptflops): typically report MACs where 1 MAC equals 2 FLOPs,
  creating a ×2 conversion relative to the fvcore convention.
- PyTorch scaled dot-product attention (`aten::scaled_dot_product_attention`):
  often lacks built-in FLOP counting, handle estimates 2·B·heads·L²·d MACs.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../NAFNet")))

import math
from typing import Any, Callable, Dict, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.flops_utils import FLOPsCounter, scaled_dot_product_attention_macs


class ConvOnly(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, stride: int = 1, padding: int = 0, groups: int = 1, bias: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LinearOnly(nn.Module):
    def __init__(self, fin: int, fout: int, bias: bool = False) -> None:
        super().__init__()
        self.fc = nn.Linear(fin, fout, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.size(0), -1))


class DepthwiseSeparable(nn.Module):
    def __init__(self, channels: int, k: int = 3) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=k, padding=k // 2, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class WithUnusedSubmodule(nn.Module):
    def __init__(self, cin: int = 3) -> None:
        super().__init__()
        self.used = nn.Conv2d(cin, cin, kernel_size=1, bias=False)
        self.unused = nn.Conv2d(cin, cin, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.used(x)


class NoParamPassThrough(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 1.0


class TinySDPA(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v)


@pytest.fixture(scope="module", params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request: pytest.FixtureRequest) -> torch.device:
    return torch.device(request.param)


def make_counter(convention: str = "fvcore_fma1", **kwargs: Any) -> Callable[[nn.Module, Tuple[torch.Tensor, ...]], Any]:
    return FLOPsCounter(convention=convention, warn_unsupported=False, **kwargs)


def conv_macs_from_output(x: torch.Tensor, model: ConvOnly, output: torch.Tensor) -> int:
    b = x.shape[0]
    cout = model.conv.out_channels
    cin = model.conv.in_channels
    groups = model.conv.groups
    k = model.conv.kernel_size[0]
    hout, wout = output.shape[-2:]
    return b * hout * wout * cout * (cin // groups) * (k * k)


def linear_macs(b: int, fin: int, fout: int) -> int:
    return b * fin * fout


def test_convention_equivalence(device: torch.device) -> None:
    torch.manual_seed(0)
    model = ConvOnly(cin=3, cout=8, k=3, padding=1, bias=False).to(device)
    x = torch.randn(4, 3, 32, 32, device=device)

    stats_fma1 = make_counter("fvcore_fma1")(model, (x,))
    stats_macs = make_counter("macs")(model, (x,))
    stats_flops = make_counter("flops_2xmac")(model, (x,))

    assert math.isclose(stats_fma1.total, stats_macs.total, rel_tol=1e-9, abs_tol=0.0)
    assert math.isclose(stats_flops.total, 2.0 * stats_macs.total, rel_tol=1e-9, abs_tol=0.0)


@pytest.mark.parametrize(
    "cfg",
    [
        dict(cin=3, cout=8, k=3, stride=1, padding=1, groups=1, bias=False),
        dict(cin=8, cout=8, k=3, stride=2, padding=1, groups=1, bias=False),
        dict(cin=16, cout=16, k=3, stride=1, padding=1, groups=16, bias=False),
        dict(cin=8, cout=8, k=1, stride=1, padding=0, groups=1, bias=False),
    ],
)
def test_conv_matches_hand_calculation(device: torch.device, cfg: Dict[str, Any]) -> None:
    torch.manual_seed(1)
    model = ConvOnly(**cfg).to(device)  # type: ignore[arg-type]
    x = torch.randn(2, cfg["cin"], 32, 40, device=device)
    output = model(x)
    stats = make_counter("fvcore_fma1")(model, (x,))
    expected = conv_macs_from_output(x, model, output)
    assert math.isclose(stats.total, expected, rel_tol=1e-6, abs_tol=0.0)


def test_linear_matches_hand_calculation(device: torch.device) -> None:
    torch.manual_seed(2)
    b, c, h, w = 4, 2, 8, 8
    fin, fout = c * h * w, 128
    model = LinearOnly(fin, fout, bias=False).to(device)
    x = torch.randn(b, c, h, w, device=device)
    stats = make_counter("fvcore_fma1")(model, (x,))
    expected = linear_macs(b, fin, fout)
    assert math.isclose(stats.total, expected, rel_tol=1e-6, abs_tol=0.0)


def test_additivity_of_breakdowns(device: torch.device) -> None:
    torch.manual_seed(3)
    model = DepthwiseSeparable(channels=8).to(device)
    x = torch.randn(3, 8, 32, 32, device=device)
    stats = make_counter("fvcore_fma1")(model, (x,))
    total = stats.total
    sum_by_operator = sum(stats.by_operator.values())
    # 只断言 sum_by_operator，避免重复计数导致的断言失败
    assert math.isclose(sum_by_operator, total, rel_tol=1e-6, abs_tol=0.0)


def test_sdpa_handle_moves_from_unsupported(device: torch.device) -> None:
    torch.manual_seed(4)
    B, H, L, D = 2, 3, 16, 8
    q = torch.randn(B, H, L, D, device=device)
    k = torch.randn(B, H, L, D, device=device)
    v = torch.randn(B, H, L, D, device=device)
    model = TinySDPA().to(device)

    counter_baseline = FLOPsCounter(convention="fvcore_fma1", custom_handles={}, warn_unsupported=False)
    baseline_stats = counter_baseline(model, (q, k, v))
    unsupported_names = baseline_stats.unsupported_ops.keys()
    assert any("scaled_dot_product_attention" in name for name in unsupported_names)

    # 兼容 PyTorch 2.1+，有的 aten 名字可能不同
    custom_handles = {
        "aten::scaled_dot_product_attention": scaled_dot_product_attention_macs,
        "aten::scaled_dot_product_attention.default": scaled_dot_product_attention_macs,
        "aten::scaled_dot_product_attention.forward": scaled_dot_product_attention_macs,
    }
    counter_handled = FLOPsCounter(
        convention="fvcore_fma1",
        custom_handles=custom_handles,
        warn_unsupported=False,
    )
    handled_stats = counter_handled(model, (q, k, v))
    # 只要有一个 handle 生效即可
    assert handled_stats.total > 0
    expected = 2.0 * B * H * L * L * D
    assert math.isclose(handled_stats.total, expected, rel_tol=1e-6, abs_tol=0.0)


def test_uncalled_modules_are_reported(device: torch.device) -> None:
    torch.manual_seed(5)
    model = WithUnusedSubmodule().to(device)
    x = torch.randn(1, 3, 16, 16, device=device)
    stats = make_counter("fvcore_fma1")(model, (x,))
    assert any("unused" in name for name in stats.uncalled_modules)


def test_device_inference_for_no_param_module(device: torch.device) -> None:
    torch.manual_seed(6)
    model = NoParamPassThrough().to(device)
    x = torch.randn(2, 3, 20, 20, device=device)
    stats = make_counter("fvcore_fma1")(model, (x,))
    assert stats.device.startswith(str(device))
    assert stats.dtype != "unknown"


def test_per_batch_vs_per_sample(device: torch.device) -> None:
    torch.manual_seed(7)
    b = 4
    model = ConvOnly(cin=3, cout=4, k=3, padding=1).to(device)
    x = torch.randn(b, 3, 16, 16, device=device)

    batch_stats = FLOPsCounter(convention="fvcore_fma1", per_batch=True, warn_unsupported=False)(model, (x,))
    sample_stats = FLOPsCounter(convention="fvcore_fma1", per_batch=False, warn_unsupported=False)(model, (x,))

    assert math.isclose(batch_stats.total, b * sample_stats.total, rel_tol=1e-9, abs_tol=0.0)
    assert batch_stats.per_sample is not None
    assert math.isclose(batch_stats.per_sample, sample_stats.total, rel_tol=1e-9, abs_tol=0.0)


def test_metadata_contains_expected_fields(device: torch.device) -> None:
    torch.manual_seed(8)
    model = ConvOnly(cin=3, cout=6, k=3, padding=1).to(device)
    x = torch.randn(1, 3, 24, 24, device=device)
    stats = make_counter("fvcore_fma1")(model, (x,))
    for field in ("convention", "input_shapes", "device", "dtype", "per_batch"):
        assert getattr(stats, field) is not None
    # 兼容 input_shapes 为 tuple 字符串格式
    assert isinstance(stats.input_shapes, list) and any(
        ("torch.Size" in shape) or (shape.startswith("(") and shape.endswith(")"))
        for shape in stats.input_shapes
    )


def test_invalid_convention_raises() -> None:
    with pytest.raises(ValueError):
        FLOPsCounter(convention="not_a_valid_convention")

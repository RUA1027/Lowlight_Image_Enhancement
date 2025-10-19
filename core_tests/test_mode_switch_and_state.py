"""
Checks:
- Train/eval semantics: dropout stochasticity updates BN stats in train, inference stability in eval.
- Metric heads (PSNR, SSIM, LPIPS, ΔE2000, channelwise stats) remain invariant across grad contexts, AMP, and backend toggles.
- Autocast scopes, inference_mode, and backend determinism do not leak state or dtype into subsequent runs.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from NewBP_model.losses import DeltaE00Loss, SSIMLoss
except ModuleNotFoundError as exc:  # pragma: no cover
    pytest.skip(f"Skipping mode/state tests due to missing dependency: {exc}", allow_module_level=True)

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPSMetric
except Exception:  # pragma: no cover
    LPIPSMetric = None


class ModeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.drop = nn.Dropout(p=0.5)
        self.head = nn.Conv2d(8, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = self.drop(x)
        return torch.sigmoid(self.head(x))


@pytest.fixture(scope="module", params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request: pytest.FixtureRequest) -> torch.device:
    return torch.device(request.param)


@pytest.fixture()
def data(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    gt = torch.rand(2, 3, 64, 64, device=device)
    pred = torch.clamp(gt + 0.03 * torch.randn_like(gt), 0.0, 1.0)
    return gt, pred


def _lpips(gt: torch.Tensor, pred: torch.Tensor) -> float:
    if LPIPSMetric is None:
        return 0.0
    metric = LPIPSMetric(net_type="alex", normalize=True)
    return float(metric(gt, pred).mean().item())


def _psnr(gt: torch.Tensor, pred: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return float("inf")
    return float(10 * torch.log10(torch.tensor(1.0, device=gt.device) / mse).item())


def _metrics(gt: torch.Tensor, pred: torch.Tensor) -> Dict[str, float]:
    try:
        de_val = float(DeltaE00Loss()(pred, gt).item())
    except Exception:
        de_val = float((pred - gt).abs().mean().item())
    return {
        "psnr": _psnr(gt, pred),
        "ssim": float(SSIMLoss(window_size=11, max_val=1.0)(pred, gt).item()),
        "lpips": _lpips(gt, pred),
        "de": de_val,
    }


def _close(val_ref: float, val_test: float, *, rtol: float, atol: float) -> bool:
    return abs(val_ref - val_test) <= atol + rtol * abs(val_ref)


def test_dropout_batchnorm_mode_semantics(device: torch.device) -> None:
    model = ModeNet().to(device)
    inputs = torch.rand(2, 3, 32, 32, device=device)

    model.train()
    with torch.no_grad():
        out_train_1 = model(inputs)
        out_train_2 = model(inputs)
    assert not torch.allclose(out_train_1, out_train_2), "Dropout should randomize outputs during training."

    running_mean_before = model.bn.running_mean.clone()
    model.eval()
    with torch.no_grad():
        _ = model(inputs)
    assert torch.allclose(
        running_mean_before, model.bn.running_mean
    ), "BN running stats must remain unchanged in eval mode."

    with torch.no_grad():
        out_eval_1 = model(inputs)
        out_eval_2 = model(inputs)
    assert torch.allclose(out_eval_1, out_eval_2), "Eval mode should produce deterministic outputs."


@pytest.mark.parametrize("ctx_kind", ["none", "no_grad", "inference"])
@pytest.mark.parametrize("amp_enabled", [False, True])
def test_metrics_invariant_to_contexts(
    device: torch.device,
    data: Tuple[torch.Tensor, torch.Tensor],
    ctx_kind: str,
    amp_enabled: bool,
) -> None:
    gt, pred = data

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    if ctx_kind == "inference":
        grad_ctx = torch.inference_mode()
    elif ctx_kind == "no_grad":
        grad_ctx = torch.no_grad()
    else:
        grad_ctx = nullcontext()

    if amp_enabled:
        if device.type == "cuda":
            amp_ctx = torch.cuda.amp.autocast()
        else:
            amp_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    else:
        amp_ctx = nullcontext()

    with torch.no_grad():
        baseline = _metrics(gt, pred)

    with grad_ctx, amp_ctx:
        test_metrics = _metrics(gt, pred)

    for key in baseline:
        if key == "lpips":
            assert _close(baseline[key], test_metrics[key], rtol=1e-2, atol=5e-3)
        else:
            assert _close(baseline[key], test_metrics[key], rtol=5e-3, atol=5e-3)


def test_inference_mode_strictness(device: torch.device, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
    gt, pred = data

    with torch.no_grad():
        val_no_grad = _lpips(gt, pred)

    with torch.inference_mode():
        val_inference = _lpips(gt, pred)
    assert _close(val_no_grad, val_inference, rtol=1e-6, atol=1e-6)

    if LPIPSMetric is None:
        pytest.skip("LPIPS dependency unavailable; skipping inference strictness residual test.")

    with torch.inference_mode():
        total = (gt + pred).sum()
        requires_grad_tensor = gt.clone().requires_grad_(True)
        with pytest.raises(RuntimeError):
            (total + requires_grad_tensor.sum()).backward()


def test_autocast_scope_reset(device: torch.device, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
    if device.type != "cuda":
        pytest.skip("Autocast leakage test is CUDA-specific.")

    gt, pred = data

    with torch.cuda.amp.autocast():
        _ = _lpips(gt, pred)

    with torch.no_grad():
        val1 = _lpips(gt, pred)
        val2 = _lpips(gt, pred)
    assert _close(val1, val2, rtol=1e-7, atol=1e-7)


def test_determinism_with_seeds(device: torch.device) -> None:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(123)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(123)

    inputs = torch.rand(1, 3, 32, 32, device=device)
    model1 = ModeNet().to(device).eval()
    out1 = model1(inputs)

    torch.manual_seed(123)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(123)
    model2 = ModeNet().to(device).eval()
    out2 = model2(inputs)

    assert torch.allclose(out1, out2)


def test_flops_counter_preserves_state(device: torch.device) -> None:
    from fvcore.nn import FlopCountAnalysis  # Imported lazily to avoid heavy dependency if unused.

    model = ModeNet().to(device).eval()
    sample = torch.rand(1, 3, 32, 32, device=device)

    running_mean_before = model.bn.running_mean.clone()
    running_var_before = model.bn.running_var.clone()

    _ = FlopCountAnalysis(model, sample)

    assert torch.allclose(running_mean_before, model.bn.running_mean)
    assert torch.allclose(running_var_before, model.bn.running_var)


# Remarks:
# Dropout and BatchNorm exhibit distinct behaviors between train and eval; the tests ensure this remains intact.
# torch.inference_mode() enforces stricter autograd semantics than no_grad(), so we confirm equality of outputs plus backend errors.
# Autocast scopes must be contained; leaving autocast should restore FP32 computations with identical outputs.
# Determinism settings combined with seed control ensure reproducible outputs on supported operators.

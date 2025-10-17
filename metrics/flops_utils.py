"""FLOP/MAC analysis utilities (model-agnostic, evaluation oriented).

The counter in this module relies on ``fvcore.FlopCountAnalysis`` which counts
one fused multiply-add (FMA) as **one** FLOP. To make comparisons explicit, the
results can be reported in three interchangeable conventions:

* ``fvcore_fma1`` – fvcore default (1×FMA = 1×FLOP).
* ``macs`` – numerically identical, emphasising multiply-accumulate counts.
* ``flops_2xmac`` – some tools/papers double-count MACs (1×MAC = 2×FLOPs).

Reference formulas (bias omitted) to sanity-check the numbers:

* Conv2d: ``MACs = B * H_out * W_out * C_out * (C_in / groups * k_h * k_w)``.
* Linear: ``MACs = B * in_features * out_features``.

Every report contains total counts, operator/module breakdowns, unsupported
operators, uncalled modules, input shape, spatial resolution, batch size,
dtype, device, convention, and any registered custom operator handles so that
control and experiment models can be audited under exactly the same settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import warnings

import torch
from fvcore.nn import FlopCountAnalysis

TensorLike = Union[torch.Tensor, Any]
InputsType = Tuple[TensorLike, ...]


def _ensure_tuple(inputs: Union[TensorLike, Sequence[TensorLike], InputsType]) -> InputsType:
    if isinstance(inputs, tuple):
        return inputs
    if isinstance(inputs, Sequence) and not isinstance(inputs, (torch.Tensor, str, bytes)):
        return tuple(inputs)
    return (inputs,)


def _infer_device(model: torch.nn.Module, inputs: InputsType) -> Optional[torch.device]:
    for param in model.parameters():
        return param.device
    for buffer in model.buffers():
        return buffer.device
    for item in inputs:
        if isinstance(item, torch.Tensor):
            return item.device
    return None


def _infer_dtype(model: torch.nn.Module, inputs: InputsType) -> Optional[torch.dtype]:
    for param in model.parameters():
        return param.dtype
    for buffer in model.buffers():
        return buffer.dtype
    for item in inputs:
        if isinstance(item, torch.Tensor):
            return item.dtype
    return None


def _first_tensor(inputs: InputsType) -> Optional[torch.Tensor]:
    for item in inputs:
        if isinstance(item, torch.Tensor):
            return item
    return None


def _convert_convention(value_fma1: float, convention: str) -> float:
    if convention in {"fvcore_fma1", "macs"}:
        return value_fma1
    if convention == "flops_2xmac":
        return 2.0 * value_fma1
    raise ValueError(f"Unsupported convention '{convention}'.")


def _batch_size(inputs: InputsType) -> Optional[int]:
    tensor = _first_tensor(inputs)
    if tensor is None or tensor.ndim == 0:
        return None
    return int(tensor.shape[0])


def _spatial_resolution(inputs: InputsType) -> Optional[Tuple[int, ...]]:
    tensor = _first_tensor(inputs)
    if tensor is None or tensor.ndim < 3:
        return None
    return tuple(int(dim) for dim in tensor.shape[-2:])


def _normalize_inputs(inputs: InputsType, device: torch.device) -> InputsType:
    normalized: List[TensorLike] = []
    for item in inputs:
        if isinstance(item, torch.Tensor) and item.device != device:
            normalized.append(item.to(device))
        else:
            normalized.append(item)
    return tuple(normalized)


def _operator_dict_to_float(mapping: Mapping[str, Union[float, torch.Tensor]]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key, value in mapping.items():
        if isinstance(value, torch.Tensor):
            result[key] = float(value.item())
        else:
            result[key] = float(value)
    return result


def _module_operator_dict_to_float(
    mapping: Mapping[str, Mapping[str, Union[float, torch.Tensor]]]
) -> Dict[str, Dict[str, float]]:
    converted: Dict[str, Dict[str, float]] = {}
    for module_name, counter in mapping.items():
        converted[module_name] = {}
        for op_name, value in counter.items():
            converted[module_name][op_name] = float(value.item() if isinstance(value, torch.Tensor) else value)
    return converted


def _dtype_string(model: torch.nn.Module, inputs: InputsType) -> str:
    dtype = _infer_dtype(model, inputs)
    return str(dtype) if dtype is not None else "unknown"


@dataclass
class FLOPsResult:
    total: float
    by_operator: Dict[str, float]
    by_module: Dict[str, float]
    by_module_and_operator: Dict[str, Dict[str, float]]
    unsupported_ops: Dict[str, int]
    uncalled_modules: List[str]
    convention: str
    input_shapes: List[str]
    batch_size: Optional[int]
    spatial_resolution: Optional[Tuple[int, ...]]
    per_batch: bool
    per_sample: Optional[float]
    dtype: str
    device: str
    custom_handles: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "by_operator": self.by_operator,
            "by_module": self.by_module,
            "by_module_and_operator": self.by_module_and_operator,
            "unsupported_ops": self.unsupported_ops,
            "uncalled_modules": self.uncalled_modules,
            "convention": self.convention,
            "input_shape": self.input_shapes,
            "batch_size": self.batch_size,
            "spatial_resolution": self.spatial_resolution,
            "per_batch": self.per_batch,
            "per_sample": self.per_sample,
            "dtype": self.dtype,
            "device": self.device,
            "custom_handles": self.custom_handles,
        }


class FLOPsCounter:
    """
    Generic FLOP/MAC counter built on top of ``fvcore.FlopCountAnalysis``.

    Parameters
    ----------
    convention:
        One of ``'fvcore_fma1'``, ``'macs'``, ``'flops_2xmac'``.
    custom_handles:
        Optional mapping from operator name to handle function, allowing custom
        accounting for attention/einsum/etc.
    warn_unsupported:
        Whether to emit fvcore warnings for unsupported operators / uncalled modules.
    per_batch:
        If ``True`` (default) totals are reported per batch; otherwise they are
        normalised by batch size (per-sample view).
    device:
        Force the analysis to run on a specific device; otherwise the counter
        infers it from model parameters/buffers or the input tensors.
    """

    def __init__(
        self,
        *,
        convention: str = "fvcore_fma1",
        custom_handles: Optional[Mapping[str, Any]] = None,
        warn_unsupported: bool = True,
        per_batch: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if convention not in {"fvcore_fma1", "macs", "flops_2xmac"}:
            raise ValueError("`convention` must be one of {'fvcore_fma1','macs','flops_2xmac'}.")
        self.convention = convention
        self.custom_handles = dict(custom_handles) if custom_handles else {}
        self.warn_unsupported = warn_unsupported
        self.per_batch = per_batch
        self.device = torch.device(device) if device is not None else None

    def register_handle(self, op_name: str, handle_fn: Any) -> None:
        """Register or overwrite a custom operator handle."""

        self.custom_handles[op_name] = handle_fn

    @torch.no_grad()
    def __call__(self, model: torch.nn.Module, inputs: Union[TensorLike, InputsType]) -> FLOPsResult:
        was_training = model.training
        model.eval()

        normalized_inputs = _ensure_tuple(inputs)
        tensor_inputs = [item for item in normalized_inputs if isinstance(item, torch.Tensor)]
        input_devices = {tensor.device for tensor in tensor_inputs}

        chosen_device = self.device or _infer_device(model, normalized_inputs)
        if not input_devices and chosen_device is None:
            chosen_device = torch.device("cpu")
        elif chosen_device is None:
            chosen_device = next(iter(input_devices))
        elif input_devices and chosen_device not in input_devices:
            warnings.warn(
                f"Model parameters on {chosen_device}, inputs on {input_devices}; using input device for analysis.",
                RuntimeWarning,
            )
            chosen_device = next(iter(input_devices))

        if input_devices and len(input_devices) > 1:
            warnings.warn(
                f"Input tensors span multiple devices {input_devices}; "
                f"moving all to {chosen_device} for FLOPs analysis.",
                RuntimeWarning,
            )

        normalized_inputs = _normalize_inputs(normalized_inputs, chosen_device)

        original_device = _infer_device(model, ())
        moved_model = False
        if original_device is not None and chosen_device != original_device:
            warnings.warn(
                f"Moving model from {original_device} to {chosen_device} for FLOPs analysis.",
                RuntimeWarning,
            )
            model.to(chosen_device)
            moved_model = True
        elif original_device is None and tensor_inputs:
            model.to(chosen_device)
            moved_model = True

        analysis = FlopCountAnalysis(model, normalized_inputs)
        analysis.unsupported_ops_warnings(self.warn_unsupported)
        analysis.uncalled_modules_warnings(self.warn_unsupported)

        for name, handle in self.custom_handles.items():
            analysis.set_op_handle(name, handle)

        try:
            total_fma1 = float(analysis.total())
        finally:
            if moved_model and original_device is not None and original_device != chosen_device:
                model.to(original_device)
            if was_training:
                model.train()

        if not self.per_batch:
            bs = _batch_size(normalized_inputs) or 1
            total_fma1 /= float(bs)

        by_operator = _operator_dict_to_float(analysis.by_operator())
        by_module = _operator_dict_to_float(analysis.by_module())
        by_mod_and_op = _module_operator_dict_to_float(analysis.by_module_and_operator())

        total = _convert_convention(total_fma1, self.convention)
        by_operator = {k: _convert_convention(v, self.convention) for k, v in by_operator.items()}
        by_module = {k: _convert_convention(v, self.convention) for k, v in by_module.items()}
        by_mod_and_op = {
            module: {op: _convert_convention(val, self.convention) for op, val in ops.items()}
            for module, ops in by_mod_and_op.items()
        }

        batch = _batch_size(normalized_inputs)
        per_sample = None
        if self.per_batch:
            if batch and batch > 0:
                per_sample = total / float(batch)
        else:
            per_sample = total

        input_shapes = [
            tuple(item.shape) if isinstance(item, torch.Tensor) else repr(type(item))
            for item in normalized_inputs
        ]

        return FLOPsResult(
            total=total,
            by_operator=by_operator,
            by_module=by_module,
            by_module_and_operator=by_mod_and_op,
            unsupported_ops={name: int(count) for name, count in analysis.unsupported_ops().items()},
            uncalled_modules=list(analysis.uncalled_modules()),
            convention=self.convention,
            input_shapes=[str(shape) for shape in input_shapes],
            batch_size=batch,
            spatial_resolution=_spatial_resolution(normalized_inputs),
            per_batch=self.per_batch,
            per_sample=per_sample,
            dtype=_dtype_string(model, normalized_inputs),
            device=str(chosen_device),
            custom_handles=tuple(self.custom_handles.keys()),
        )


def scaled_dot_product_attention_macs(inputs: Sequence[Any], outputs: Any) -> float:
    """
    Approximate MACs for ``aten::scaled_dot_product_attention`` (FMA=1 units).

    Assumes query/key/value tensors with shape ``[B, heads, L, d_k]``. Two
    dominant matrix multiplications contribute:
        Q @ K^T  → ``B * heads * L * L * d_k``
        (QK^T) @ V → same again
    Total ≈ ``2 * B * heads * L * L * d_k``.
    """

    if not inputs:
        return 0.0
    query = inputs[0]
    if not isinstance(query, torch.Tensor) or query.ndim < 4:
        return 0.0
    batch, heads, length, d_k = query.shape[:4]
    return 2.0 * float(batch) * float(heads) * float(length) * float(length) * float(d_k)


def count_flops(model: torch.nn.Module, input_tensor: torch.Tensor, unit: str = "G") -> float:
    """
    Backward-compatible helper returning FLOPs in the requested unit (default G).

    Args:
        model: PyTorch module to analyse.
        input_tensor: Example input matching real inference usage.
        unit: ``"M"`` (mega) or ``"G"`` (giga).
    """

    if unit not in {"M", "G"}:
        raise ValueError("`unit` must be 'M' or 'G'.")

    counter = FLOPsCounter(convention="fvcore_fma1", per_batch=True)
    stats = counter(model, (input_tensor,))
    total_fma1 = stats.total

    scale = 1e6 if unit == "M" else 1e9
    return float(total_fma1) / scale


if __name__ == "__main__":  # pragma: no cover - demonstration only
    import torch.nn as nn

    class Demo(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(16 * 64 * 64, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu(self.conv(x))
            return self.fc(x.flatten(1))

    model = Demo()
    inputs = (torch.randn(4, 3, 64, 64),)  # batch size 4
    counter = FLOPsCounter(convention="fvcore_fma1", per_batch=True)
    stats = counter(model, inputs).as_dict()

    total_g = stats["total"] / 1e9
    per_sample_g = (stats["per_sample"] / 1e9) if stats["per_sample"] is not None else None
    per_sample_str = f"{per_sample_g:.4f} G" if per_sample_g is not None else "n/a"
    print(
        f"[FLOPs] total={total_g:.4f} G ({stats['convention']}), "
        f"batch={stats['batch_size']}, per_sample={per_sample_str}, "
        f"input={stats['input_shape']}, device={stats['device']}"
    )
    print("Top operators:")
    for name, value in sorted(stats["by_operator"].items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {name}: {value / 1e9:.4f} G")
    if stats["unsupported_ops"]:
        print("Unsupported ops:", stats["unsupported_ops"])

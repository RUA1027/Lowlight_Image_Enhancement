"""Unified evaluation pipeline for low-light restoration models.

This script aggregates all metric utilities into a single entry point so that multiple
models (e.g., NewBP-NAFNet, vanilla NAFNet, U-Net, SwinIR, etc.) can be evaluated under
the same criteria in environments such as Google Colab.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch

from metrics.psnr import PSNRMetric
from metrics.ssim import SSIMMetric
from metrics.lpips_metric import LPIPSEvaluator
from metrics.parameter_utils import count_parameters
from metrics.flops_utils import count_flops
from metrics.inference_time import measure_inference_time


def run_full_evaluation(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: torch.device | str,
    *,
    data_range: float = 1.0,
    ssim_win_size: int = 7,
    lpips_net: str = "alex",
    compute_flops: bool = False,
    flops_unit: str = "G",
    flops_sample: Optional[torch.Tensor] = None,
    benchmark_speed: bool = False,
    warmup_runs: int = 20,
    timed_runs: int = 100,
) -> Dict[str, Any]:
    """Evaluate a restoration model on a dataset and return aggregated metrics.

    Args:
        model: Any ``torch.nn.Module`` that maps noisy images to restored images.
        dataloader: Iterable yielding ``(noisy, clean)`` tensors.
        device: Target device (``"cuda"`` or ``"cpu"``). Model and data are moved here.
        data_range: Maximum pixel value in the clean references (default: 1.0 for normed data).
        ssim_win_size: Window size used by SSIM (must be odd).
        lpips_net: Backbone for LPIPS (``"alex"`` or ``"vgg"``). See ``lpips`` docs.
        compute_flops: If ``True``, compute FLOPs using ``count_flops``.
        flops_unit: Unit for FLOPs (``"M"`` or ``"G"``). Ignored if ``compute_flops`` is ``False``.
        flops_sample: Optional tensor whose shape matches actual inference input. If ``None``,
            the first batch from ``dataloader`` is used automatically (single sample taken).
        benchmark_speed: If ``True`` and ``device`` is CUDA-enabled, run ``measure_inference_time``.
        warmup_runs: Number of warmup iterations for inference benchmarking.
        timed_runs: Number of timed iterations for inference benchmarking.

    Returns:
        Dictionary containing aggregated statistics for PSNR, SSIM, LPIPS, parameter counts,
        and optionally FLOPs/inference latency.
    """

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    psnr_metric = PSNRMetric(data_range=data_range)
    ssim_metric = SSIMMetric(data_range=data_range, win_size=ssim_win_size)
    lpips_device = device.type if device.type == "cuda" else "cpu"
    lpips_eval = LPIPSEvaluator(net=lpips_net, device=lpips_device)
    lpips_eval.reset()

    sample_for_flops = flops_sample

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                noisy, clean = batch[0], batch[1]
            else:
                raise ValueError(
                    "Dataloader must yield (noisy, clean) pairs. Received type: "
                    f"{type(batch)}"
                )

            noisy = noisy.to(device)
            clean = clean.to(device)

            restored = model(noisy)

            psnr_metric.update(clean, restored)
            ssim_metric.update(clean, restored)
            lpips_eval.update(clean, restored)

            if sample_for_flops is None:
                # Clone to detach from graph and ensure batch size of 1 for FLOPs/latency.
                sample_for_flops = noisy[:1].clone().detach()

    results: Dict[str, Any] = {
        "psnr": psnr_metric.compute(),
        "ssim": ssim_metric.compute(),
        "lpips": lpips_eval.compute(),
        "parameters": {
            "raw": count_parameters(model, unit="raw"),
            "M": count_parameters(model, unit="M"),
        },
    }

    if compute_flops:
        if sample_for_flops is None:
            raise ValueError(
                "`compute_flops` is True but no sample input was found. Provide `flops_sample`"
                " or ensure the dataloader produced at least one batch."
            )
        results["flops"] = {
            flops_unit: count_flops(model, sample_for_flops, unit=flops_unit)
        }

    if benchmark_speed:
        if device.type != "cuda":
            raise RuntimeError("`benchmark_speed` requires a CUDA device.")
        if sample_for_flops is None:
            raise ValueError("Provide a sample input for latency measurement.")
        results["inference_time_ms"] = measure_inference_time(
            model,
            sample_for_flops,
            num_warmup=warmup_runs,
            num_runs=timed_runs,
        )

    return results


if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    class TinyRestorationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(16, 3, kernel_size=3, padding=1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    torch.manual_seed(0)
    dummy_noisy = torch.rand(8, 3, 64, 64)
    dummy_clean = torch.rand(8, 3, 64, 64)
    dataset = TensorDataset(dummy_noisy, dummy_clean)
    loader = DataLoader(dataset, batch_size=2)

    model = TinyRestorationModel()

    metrics = run_full_evaluation(
        model=model,
        dataloader=loader,
        device="cpu",
        data_range=1.0,
        compute_flops=False,  # Set to True when you want FLOPs (requires sample input on same device)
        benchmark_speed=False,  # Set to True on CUDA to measure latency
    )

    for key, value in metrics.items():
        print(f"{key}: {value}")

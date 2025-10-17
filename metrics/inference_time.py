"""GPU inference benchmarking utilities for diverse restoration models."""

from __future__ import annotations

import torch


def _resolve_model_device(model: torch.nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    for buffer in model.buffers():
        return buffer.device
    raise RuntimeError("Model has no parameters or buffers to infer device from.")


def measure_inference_time(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 20,
    num_runs: int = 100,
) -> float:
    """Benchmark average per-image inference time on CUDA hardware.

    Args:
        model: PyTorch model to benchmark (will be set to eval mode during timing).
        input_tensor: Sample input tensor on the target CUDA device with realistic shape.
        num_warmup: Number of warm-up iterations prior to timed runs (default: 20).
        num_runs: Number of timed iterations (default: 100).

    Returns:
        Average inference latency per image in milliseconds.

    Raises:
        RuntimeError: If CUDA is unavailable or the model/input reside on different devices.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("measure_inference_time requires a CUDA-enabled PyTorch installation.")

    if input_tensor.device.type != "cuda":
        raise RuntimeError("`input_tensor` must be located on a CUDA device.")

    model_device = _resolve_model_device(model)
    if model_device != input_tensor.device:
        raise RuntimeError("Model and input tensor must reside on the same CUDA device.")

    was_training = model.training
    model.eval()

    torch.cuda.synchronize()

    try:
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(input_tensor)

            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            current_stream = torch.cuda.current_stream()

            start_event.record(current_stream)
            for _ in range(num_runs):
                _ = model(input_tensor)
            end_event.record(current_stream)

            torch.cuda.synchronize()
            total_time_ms = start_event.elapsed_time(end_event)
    finally:
        if was_training:
            model.train()

    batch_size = input_tensor.shape[0]
    total_images = num_runs * batch_size
    return total_time_ms / total_images


if __name__ == "__main__":
    import torch.nn as nn

    if not torch.cuda.is_available():
        print("This benchmark function requires a CUDA-enabled GPU.")
    else:
        device = torch.device("cuda")
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=7),
        ).to(device)

        dummy_input = torch.randn(1, 3, 512, 512, device=device)

        avg_time = measure_inference_time(model, dummy_input)

        print("--- Model Inference Speed Benchmark ---")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Hardware: {torch.cuda.get_device_name(0)}")
        print(f"Average inference time per image: {avg_time:.4f} ms")

#!/usr/bin/env python
"""
Quick smoke test for HybridLossPlus and related helpers.

Usage:
    python tools/debug_losses.py --device cpu
    python tools/debug_losses.py --device cuda --steps 5 --height 128 --width 128
"""

from __future__ import annotations

import argparse
from typing import Tuple

import torch

from NewBP_model.losses import HybridLossPlus
from NewBP_model.newbp_net_arch import create_crosstalk_psf


def build_tensors(
    batch: int,
    channels: int,
    height: int,
    width: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bhat = torch.rand(batch, channels, height, width, device=device)
    b = torch.rand_like(bhat)
    a = torch.rand_like(bhat)
    return bhat, b, a


def run_checks(device: torch.device, steps: int, height: int, width: int) -> None:
    psf_mono = create_crosstalk_psf(psf_mode="mono", kernel_spec="P2").to(device)
    psf_rgb = create_crosstalk_psf(psf_mode="rgb", kernel_spec="B2").to(device)

    loss_mono = HybridLossPlus(
        device=device.type,
        w_l1_raw=1.0,
        w_perc=0.02,
        w_deltaE=0.02,
        w_phys=0.1,
        use_deltaE=True,
        use_ssim=False,
        use_lpips=False,
        use_phys=True,
        physics_psf_module=psf_mono,
    ).to(device)

    loss_rgb = HybridLossPlus(
        device=device.type,
        w_l1_raw=1.0,
        w_perc=0.02,
        w_deltaE=0.02,
        w_phys=0.1,
        use_deltaE=True,
        use_ssim=False,
        use_lpips=False,
        use_phys=True,
        physics_psf_module=psf_rgb,
    ).to(device)

    for step in range(1, steps + 1):
        bhat, b, a = build_tensors(batch=2, channels=3, height=height, width=width, device=device)
        expo_ratio = torch.full((2,), 10.0, device=device)

        bhat.requires_grad_(True)
        loss_val, logs = loss_mono(
            Bhat_raw=bhat,
            B_raw=b,
            A_raw=a,
            expo_ratio=expo_ratio,
            Bhat_srgb01=bhat.clamp(0, 1),
            B_srgb01=b.clamp(0, 1),
            A_srgb01=a.clamp(0, 1),
        )
        if not torch.isfinite(loss_val):
            raise RuntimeError(f"Mono PSF loss 出现非有限值 (step={step}): {logs}")
        loss_val.backward()
        grad_norm = bhat.grad.detach().norm().item()
        print(f"[mono] step={step} loss={loss_val.item():.6f} grad_norm={grad_norm:.6f}")

        bhat2, b2, a2 = build_tensors(batch=2, channels=3, height=height, width=width, device=device)
        expo_ratio2 = torch.full((2,), 8.0, device=device)
        bhat2.requires_grad_(True)
        loss_val_rgb, logs_rgb = loss_rgb(
            Bhat_raw=bhat2,
            B_raw=b2,
            A_raw=a2,
            expo_ratio=expo_ratio2,
            Bhat_srgb01=bhat2.clamp(0, 1),
            B_srgb01=b2.clamp(0, 1),
            A_srgb01=a2.clamp(0, 1),
        )
        if not torch.isfinite(loss_val_rgb):
            raise RuntimeError(f"RGB PSF loss 出现非有限值 (step={step}): {logs_rgb}")
        loss_val_rgb.backward()
        grad_norm_rgb = bhat2.grad.detach().norm().item()
        print(f"[rgb ] step={step} loss={loss_val_rgb.item():.6f} grad_norm={grad_norm_rgb:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HybridLossPlus 快速自检")
    parser.add_argument("--device", type=str, default="cpu", help="cpu 或 cuda")
    parser.add_argument("--steps", type=int, default=3, help="重复次数")
    parser.add_argument("--height", type=int, default=64, help="输入高度")
    parser.add_argument("--width", type=int, default=64, help="输入宽度")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device.startswith("cpu") else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，自动回退到 CPU。")
        device = torch.device("cpu")

    run_checks(device=device, steps=args.steps, height=args.height, width=args.width)
    print("所有损失项检查通过。")


if __name__ == "__main__":
    main()

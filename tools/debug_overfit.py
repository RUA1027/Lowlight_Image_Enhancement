from __future__ import annotations
#!/usr/bin/env python
# 添加项目根目录到 sys.path，确保可以导入 NewBP_model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
Rapid overfit/debug harness for the NewBP + NAFNet generator.

This script is designed to follow the "8x8 first" philosophy:
run the network on a single mini batch, confirm that loss and gradients behave,
then gradually scale up. By default it uses a tiny NAFNet variant and L1 loss
on synthetic data so it executes in seconds.

Example:
    python tools/debug_overfit.py --device cuda --iters 200 --height 64 --width 64

Switch to HybridLossPlus (with physics PSF) after the basic L1 check passes:
    python tools/debug_overfit.py --loss hybrid --enable-phys --device cuda
"""



import argparse
import math
from typing import Tuple

import torch
from torch import nn, optim

from NewBP_model.newbp_net_arch import create_newbp_net, create_crosstalk_psf
from NewBP_model.losses import HybridLossPlus

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-batch overfit harness for NewBP-NAFNet.")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda or cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--iters", type=int, default=200, help="Training iterations.")
    parser.add_argument("--height", type=int, default=64, help="Input height (must be divisible by 2**depth).")
    parser.add_argument("--width", type=int, default=64, help="Input width (must be divisible by 2**depth).")
    parser.add_argument("--batch-size", type=int, default=1, help="Mini batch size.")
    parser.add_argument("--in-channels", type=int, default=3, help="Number of input/output channels.")
    parser.add_argument("--naf-width", type=int, default=16, help="Base channel width for tiny NAFNet.")
    parser.add_argument("--depth", type=int, default=2, help="Encoder/decoder depth (number of downs/ups).")
    parser.add_argument("--middle-blocks", type=int, default=1, help="Number of middle NAF blocks.")
    parser.add_argument("--kernel-type", default="panchromatic", choices=["panchromatic", "rgb"],
                        help="PSF kernel type to match training scenario.")
    parser.add_argument("--kernel-spec", default="P2", choices=["P2", "B2"],
                        help="PSF kernel specification.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--loss", choices=["l1", "hybrid"], default="l1",
                        help="Loss to use for the overfit test.")
    parser.add_argument("--enable-phys", action="store_true",
                        help="Enable physics consistency term when using hybrid loss.")
    parser.add_argument("--enable-deltae", action="store_true",
                        help="Enable DeltaE term when using hybrid loss.")
    parser.add_argument("--log-interval", type=int, default=20, help="Logging interval.")
    return parser.parse_args()


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def build_tiny_newbp(args: argparse.Namespace, device: torch.device) -> nn.Module:
    naf_depth = max(1, args.depth)
    enc_blocks = [1] * naf_depth
    dec_blocks = [1] * naf_depth
    nafnet_params = dict(
        img_channel=args.in_channels,
        width=args.naf_width,
        enc_blk_nums=enc_blocks,
        dec_blk_nums=dec_blocks,
        middle_blk_num=max(1, args.middle_blocks),
    )
    model = create_newbp_net(
        in_channels=args.in_channels,
        kernel_type=args.kernel_type,
        kernel_spec=args.kernel_spec,
        nafnet_params=nafnet_params,
    )
    return model.to(device)


def generate_synthetic_batch(
    args: argparse.Namespace, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    dummy_input = torch.rand(args.batch_size, args.in_channels, args.height, args.width, device=device)
    dummy_target = torch.rand_like(dummy_input)
    return dummy_input, dummy_target


def build_loss(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.loss == "l1":
        return nn.L1Loss()

    psf_mode = "mono" if args.kernel_type == "panchromatic" else "rgb"
    psf = create_crosstalk_psf(psf_mode=psf_mode, kernel_spec=args.kernel_spec).to(device)
    return HybridLossPlus(
        device=device.type,
        w_l1_raw=1.0,
        w_perc=0.02,
        w_lpips=0.0,
        w_deltaE=0.02 if args.enable_deltae else 0.0,
        w_ssim=0.0,
        w_phys=0.1 if args.enable_phys else 0.0,
        use_deltaE=args.enable_deltae,
        use_ssim=False,
        use_lpips=False,
        use_phys=args.enable_phys,
        physics_psf_module=psf,
    ).to(device)


def compute_grad_norm(model: nn.Module) -> float:
    total_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            if grad.numel() == 0:
                continue
            total_sq += grad.norm(2).item() ** 2
    return math.sqrt(total_sq) if total_sq > 0 else 0.0


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device.startswith("cpu") else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")

    if args.height % (2 ** args.depth) != 0 or args.width % (2 ** args.depth) != 0:
        raise ValueError("height and width must be divisible by 2**depth to satisfy NAFNet padding rules.")

    set_seed(args.seed, device)
    model = build_tiny_newbp(args, device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = build_loss(args, device)
    dummy_input, dummy_target = generate_synthetic_batch(args, device)
    exposure_ratio = torch.ones(args.batch_size, device=device)

    best_loss = float("inf")

    print("Starting single-batch overfit test...")
    for step in range(1, args.iters + 1):
        optimizer.zero_grad(set_to_none=True)

        output = model(dummy_input)

        if args.loss == "l1":
            loss = criterion(output, dummy_target)
        else:
            loss, logs = criterion(
                Bhat_raw=output,
                B_raw=dummy_target,
                A_raw=dummy_input,
                expo_ratio=exposure_ratio,
                Bhat_srgb01=output.clamp(0, 1),
                B_srgb01=dummy_target.clamp(0, 1),
                A_srgb01=dummy_input.clamp(0, 1),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Hybrid loss produced non-finite value at step {step}. Logs: {logs}")

        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss became non-finite at step {step}: {loss.item()}")

        loss.backward()

        grad_norm = compute_grad_norm(model)
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"Gradient norm is non-finite at step {step}.")

        optimizer.step()

        best_loss = min(best_loss, loss.item())

        if step % args.log_interval == 0 or step == 1 or step == args.iters:
            print(f"Step {step:04d} | loss={loss.item():.6f} | best={best_loss:.6f} | grad_norm={grad_norm:.4f}")

    print("Overfit test complete.")
    print(f"Final loss: {loss.item():.6f} | Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()

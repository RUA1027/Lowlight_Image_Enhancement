import math
from typing import Dict, Optional, List

import torch
from torch import nn

# Required metrics
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# Optional deps
try:
    import lpips as _lpips
except Exception:  # pragma: no cover - optional dependency
    _lpips = None  # type: ignore

try:
    import kornia  # noqa: F401
    from kornia.filters import sobel as kornia_sobel  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    kornia_sobel = None  # type: ignore

# Project-local imports
from .color_error import DeltaE00Loss
from .phys_consistency import PhysicsConsistencyLoss


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB in [0, 1] to sRGB in [0, 1].

    Applies the standard IEC 61966-2-1 sRGB EOTF.
    Expects shape (..., C, H, W) with C==1 or 3.
    """
    x = x.clamp(0.0, 1.0)
    a = 0.055
    threshold = 0.0031308
    low = 12.92 * x
    high = (1 + a) * torch.pow(x.clamp(min=threshold), 1.0 / 2.4) - a
    return torch.where(x <= threshold, low, high)


def _to_3ch(x: torch.Tensor) -> torch.Tensor:
    if x.size(1) == 3:
        return x
    if x.size(1) == 1:
        return x.repeat(1, 3, 1, 1)
    # If channel count is unexpected, try to map first 3 channels
    return x[:, :3, ...]


def _norm_to_lpips_range(x: torch.Tensor) -> torch.Tensor:
    """Normalize image in [0,1] to [-1,1] for LPIPS."""
    return x.mul(2.0).sub(1.0)


def _compute_channelwise_psnr_sse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return per-channel SSE across N,H,W.

    Both inputs expected as float tensors in [0,1], shape N,C,H,W.
    """
    se = (pred - target) ** 2
    # Sum over N,H,W -> shape [C]
    return se.sum(dim=(0, 2, 3))


def _sobel_magnitude(gray: torch.Tensor) -> torch.Tensor:
    """Compute gradient magnitude using Sobel. Expects N,1,H,W in [0,1]."""
    if kornia_sobel is not None:  # prefer kornia if available
        # kornia.filters.sobel returns gradient image with shape N,C,H,W,2 (gx, gy) in recent versions.
        grad = kornia_sobel(gray)
        # Support both possible layouts (N,C,H,W,2) or (N,2,C,H,W)
        if grad.dim() == 5 and grad.size(-1) == 2:
            gx, gy = grad[..., 0], grad[..., 1]
        elif grad.dim() == 5 and grad.size(1) == 2:
            gx, gy = grad[:, 0:1, ...], grad[:, 1:2, ...]
        else:  # fallback to norm across channel if shape unexpected
            return grad.abs().sum(dim=1, keepdim=True)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
        return mag
    # Manual Sobel fallback
    device = gray.device
    dtype = gray.dtype
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    gx = torch.nn.functional.conv2d(gray, kx, padding=1)
    gy = torch.nn.functional.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


@torch.no_grad()
def compute_metrics(model: nn.Module, dataloader, device: torch.device | str) -> Dict[str, Optional[float]]:
    """Evaluate a low-light denoising model on a validation DataLoader.

    - PSNR/SSIM: linear domain using torchmetrics with data_range=1.0
    - LPIPS: sRGB domain using lpips (VGG). If lpips not installed, returns None.
    - DeltaE_00 mean/p95: using DeltaE00Loss on sRGB; flatten pixels across dataset.
    - RGB PSNR: per-channel PSNR list [R, G, B].
    - Phys-Cons MAE: using PhysicsConsistencyLoss forward.
    - Edge-DeltaE: optional; computed on pixels with high Sobel gradient magnitude (>= 90th percentile per-image).
    """
    if isinstance(device, str):
        device = torch.device(device)

    model_was_training = model.training
    model.eval()

    # Metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    lpips_fn = None
    if _lpips is not None:
        try:
            lpips_fn = _lpips.LPIPS(net='vgg').to(device)
            lpips_fn.eval()
        except Exception:
            lpips_fn = None

    deltae_fn = DeltaE00Loss().to(device)
    deltae_fn.eval()

    phys_fn = PhysicsConsistencyLoss().to(device)
    phys_fn.eval()

    # Accumulators
    lpips_sum = 0.0
    lpips_count = 0

    # Accumulate flattened ΔE values on CPU for mean/quantile
    deltae_values: List[torch.Tensor] = []
    edge_deltae_values: List[torch.Tensor] = []

    # Channelwise SSE and count for PSNR
    ch_sse = torch.zeros(3, dtype=torch.float64, device=device)
    ch_count: Optional[int] = None

    phys_sum = 0.0
    phys_count = 0

    use_autocast = device.type == 'cuda'

    for batch in dataloader:
        # Expect batch to be dict with 'lq' and 'gt'
        lq = batch['lq'].to(device, non_blocking=True).float()
        gt = batch['gt'].to(device, non_blocking=True).float()

        # Ensure shape N,C,H,W
        if lq.dim() == 3:
            lq = lq.unsqueeze(0)
        if gt.dim() == 3:
            gt = gt.unsqueeze(0)

        with torch.cuda.amp.autocast(enabled=use_autocast):
            pred = model(lq)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred = torch.clamp(pred, 0.0, 1.0)

            # Linear-domain metrics
            psnr_metric.update(pred, gt)
            ssim_metric.update(pred, gt)

            # Channelwise SSE
            # Expand to 3ch for safety if needed
            pred_3 = _to_3ch(pred)
            gt_3 = _to_3ch(gt)
            sse = _compute_channelwise_psnr_sse(pred_3, gt_3).to(dtype=torch.float64)
            ch_sse = ch_sse + sse
            if ch_count is None:
                ch_count = int(pred_3.size(0) * pred_3.size(2) * pred_3.size(3))
            else:
                ch_count += int(pred_3.size(0) * pred_3.size(2) * pred_3.size(3))

            # LPIPS on sRGB if available
            if lpips_fn is not None:
                pred_srgb = _linear_to_srgb(pred_3)
                gt_srgb = _linear_to_srgb(gt_3)
                p_in = _norm_to_lpips_range(pred_srgb)
                g_in = _norm_to_lpips_range(gt_srgb)
                lp = lpips_fn(p_in, g_in)
                # lpips returns (N,1,1,1) or (N,) tensor
                lp = lp.view(-1)
                lpips_sum += float(lp.sum().item())
                lpips_count += int(lp.numel())

            # ΔE_00 on sRGB
            pred_srgb_for_de = _linear_to_srgb(pred_3)
            gt_srgb_for_de = _linear_to_srgb(gt_3)
            de_map = deltae_fn(pred_srgb_for_de, gt_srgb_for_de)
            # Expect de_map shape N,1,H,W or N,H,W
            if de_map.dim() == 3:
                de_map = de_map.unsqueeze(1)
            deltae_values.append(de_map.reshape(-1).detach().cpu())

            # Edge-ΔE using Sobel mask on GT luminance
            # Convert to grayscale (simple average if no color transform available)
            gt_gray = gt_srgb_for_de.mean(dim=1, keepdim=True)
            grad_mag = _sobel_magnitude(gt_gray)
            # Select high-gradient pixels: >= 90th percentile per-batch
            q = torch.quantile(grad_mag.reshape(grad_mag.size(0), -1), 0.90, dim=1)
            # Build mask per-image
            mask_list = []
            for i in range(grad_mag.size(0)):
                mask_list.append((grad_mag[i : i + 1] >= q[i]).expand_as(de_map[i : i + 1]))
            edge_mask = torch.cat(mask_list, dim=0)
            de_edge = de_map[edge_mask]
            if de_edge.numel() > 0:
                edge_deltae_values.append(de_edge.detach().cpu())

            # Physics consistency MAE
            phys_val = phys_fn(pred, lq)
            # Reduce to mean absolute error
            phys_val = torch.as_tensor(phys_val)
            phys_sum += float(phys_val.detach().abs().mean().item())
            phys_count += 1

    # Compute aggregates
    results: Dict[str, Optional[float] | List[float]] = {}

    psnr_val = psnr_metric.compute().item()
    ssim_val = ssim_metric.compute().item()
    results['psnr'] = float(psnr_val)
    results['ssim'] = float(ssim_val)

    # LPIPS average
    results['lpips'] = (lpips_sum / lpips_count) if lpips_count > 0 else None

    # DeltaE stats
    if len(deltae_values) > 0:
        all_de = torch.cat(deltae_values)
        results['deltae_mean'] = float(all_de.mean().item())
        results['deltae_p95'] = float(torch.quantile(all_de, 0.95).item())
    else:
        results['deltae_mean'] = None
        results['deltae_p95'] = None

    # Edge-DeltaE stats (optional)
    if len(edge_deltae_values) > 0:
        de_edge_all = torch.cat(edge_deltae_values)
        results['edge_deltae_mean'] = float(de_edge_all.mean().item())
        results['edge_deltae_p95'] = float(torch.quantile(de_edge_all, 0.95).item())
    # else: omit keys for cleanliness

    # RGB PSNR
    rgb_psnr: List[float] = [float('nan'), float('nan'), float('nan')]
    if ch_count and ch_count > 0:
        data_range = 1.0
        for c in range(3):
            mse_c = (ch_sse[c] / ch_count).item()
            if mse_c <= 0:
                rgb_psnr[c] = float('inf')
            else:
                rgb_psnr[c] = float(10.0 * math.log10((data_range ** 2) / mse_c))
    results['rgb_psnr'] = rgb_psnr

    # Physics consistency MAE
    results['phys_mae'] = (phys_sum / phys_count) if phys_count > 0 else None

    # Restore training state
    if model_was_training:
        model.train()

    return results


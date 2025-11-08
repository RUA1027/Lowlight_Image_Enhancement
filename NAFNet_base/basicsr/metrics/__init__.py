# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from .lowlight_metrics import (
    deltae2000_mean,
    deltae2000_p95,
    edge_deltae2000_mean,
    linear_psnr,
    linear_ssim,
    lpips_distance,
)

try:  # pragma: no cover - skimage import may fail on mismatched numpy wheels
    from .psnr_ssim import (
        calculate_psnr,
        calculate_ssim,
        calculate_ssim_left,
        calculate_psnr_left,
        calculate_skimage_ssim,
        calculate_skimage_ssim_left,
    )
except Exception as psnr_import_exc:  # noqa: F841
    def _missing_psnr(*_args, **_kwargs):
        raise ImportError(
            "PSNR/SSIM metrics require scikit-image built against the current NumPy. "
            "Install a compatible version or disable these metrics."
        ) from psnr_import_exc

    calculate_psnr = calculate_ssim = calculate_ssim_left = calculate_psnr_left = (
        calculate_skimage_ssim
    ) = calculate_skimage_ssim_left = _missing_psnr

try:  # pragma: no cover - SciPy is optional in some lightweight setups
    from .niqe import calculate_niqe
except Exception as niqe_import_exc:  # noqa: F841
    def calculate_niqe(*_args, **_kwargs):
        raise ImportError(
            "NIQE metric requires SciPy built against NumPy <2.3. "
            "Install compatible SciPy/NumPy or skip NIQE in your config."
        ) from niqe_import_exc

__all__ = [
    'calculate_psnr',
    'calculate_ssim',
    'calculate_niqe',
    'calculate_ssim_left',
    'calculate_psnr_left',
    'calculate_skimage_ssim',
    'calculate_skimage_ssim_left',
    'linear_psnr',
    'linear_ssim',
    'lpips_distance',
    'deltae2000_mean',
    'deltae2000_p95',
    'edge_deltae2000_mean',
]

import os
import sys
import logging

import torch.nn as nn

NAFNET_ROOT = os.path.join(os.path.dirname(__file__), 'NAFNet')
if NAFNET_ROOT not in sys.path:
    sys.path.insert(0, NAFNET_ROOT)

from basicsr.models.archs.NAFNet_arch import NAFNet  # pyright: ignore[reportMissingImports]
from .newbp_layer import NewBPLayer, CrosstalkPSF, build_psf_kernels  # 使用相对导入


logger = logging.getLogger(__name__)


def create_newbp_net(in_channels=3, kernel_type='panchromatic', kernel_spec='P2', nafnet_params=None):
    if nafnet_params is None:
        nafnet_params = {}

    if not isinstance(nafnet_params, dict):
        raise TypeError("nafnet_params must be a dictionary if provided.")

    nafnet_config = dict(nafnet_params)
    nafnet_config.setdefault('img_channel', in_channels)

    # Scenario B invariants:
    # - Forward identity w.r.t K: A_srgb -> NAFNet -> Bhat_srgb; no conv2d(..., K) on input side.
    # - Physical consistency is enforced only in the loss path using a fixed PSF (register_buffer).
    base_model = NAFNet(**nafnet_config)

    # IMPORTANT: Do NOT convolve the input A with K in the forward.
    # We keep NAFNet intact; K is only used in the loss branch (output-side consistency).
    # NewBPLayer is no longer wired into the network input here to avoid double crosstalk.

    logger.info(
        "[NewBP-Net] Created (Scenario B: no input-side K). kernel_type='%s', kernel_spec='%s', in_channels=%s. "
        "Use CrosstalkPSF only in the loss branch.",
        kernel_type,
        kernel_spec,
        in_channels,
    )

    return base_model


def create_crosstalk_psf(psf_mode: str = 'mono', kernel_spec: str = 'P2'):
    """
    Helper to build a fixed PSF module for loss graph only.

    psf_mode: 'mono' -> single 3x3 kernel shared across channels; 'rgb' -> per-channel 3x3 kernels.
    kernel_spec: 'P2' for mono (panchromatic), 'B2' for rgb.
    """
    # map legacy names to new modes
    if psf_mode not in { 'mono', 'rgb' }:
        raise ValueError("psf_mode must be 'mono' or 'rgb'")
    kernels = build_psf_kernels(psf_mode, kernel_spec)
    return CrosstalkPSF(mode=psf_mode, kernels=kernels)

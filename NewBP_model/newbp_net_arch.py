import os
import sys
import logging
from typing import Any, Dict, Optional, Sequence

# Ensure we can import the in-repo BasicSR under NAFNet_base/basicsr
try:
    from basicsr.models.archs.NAFNet_arch import NAFNet  # type: ignore
except ModuleNotFoundError:
    _here = os.path.abspath(os.path.dirname(__file__))
    _repo_root = os.path.abspath(os.path.join(_here, '..'))
    _nafnet_base = os.path.join(_repo_root, 'NAFNet_base')
    # Prefer adding the NAFNet_base folder so that "import basicsr" resolves as a package
    for p in (_nafnet_base, os.path.join(_nafnet_base, 'basicsr'), _repo_root):
        if p not in sys.path and os.path.isdir(p):
            sys.path.insert(0, p)
    from basicsr.models.archs.NAFNet_arch import NAFNet  # type: ignore

from .newbp_layer import NewBPLayer, CrosstalkPSF, build_psf_kernels  # ʹ����Ե���


logger = logging.getLogger(__name__)


def _maybe_list(seq: Optional[Sequence[int]]) -> Optional[list[int]]:
    if seq is None:
        return None
    return list(seq)


def create_newbp_net(
    in_channels: int = 3,
    kernel_type: str = 'panchromatic',
    kernel_spec: str = 'P2',
    width: Optional[int] = None,
    enc_blk_nums: Optional[Sequence[int]] = None,
    middle_blk_num: Optional[int] = None,
    dec_blk_nums: Optional[Sequence[int]] = None,
    nafnet_params: Optional[Dict[str, Any]] = None,
    **nafnet_kwargs: Any,
):
    if nafnet_params is not None and not isinstance(nafnet_params, dict):
        raise TypeError("nafnet_params must be a dictionary if provided.")

    nafnet_config: Dict[str, Any] = {}
    if nafnet_params:
        nafnet_config.update(nafnet_params)

    # Allow callers to pass additional NAFNet kwargs directly.
    if nafnet_kwargs:
        nafnet_config.update(nafnet_kwargs)

    # Map the public API arguments to the underlying NAFNet configuration.
    nafnet_config['img_channel'] = in_channels
    if width is not None:
        nafnet_config['width'] = width
    if enc_blk_nums is not None:
        nafnet_config['enc_blk_nums'] = _maybe_list(enc_blk_nums)
    if middle_blk_num is not None:
        nafnet_config['middle_blk_num'] = middle_blk_num
    if dec_blk_nums is not None:
        nafnet_config['dec_blk_nums'] = _maybe_list(dec_blk_nums)

    # Scenario B invariants:
    # - Forward identity w.r.t K: A_srgb -> NAFNet -> Bhat_srgb; no conv2d(..., K) on input side.
    # - Physical consistency is enforced only in the loss path using a fixed PSF (register_buffer).
    base_model = NAFNet(**nafnet_config)

    # IMPORTANT: Do NOT convolve the input A with K in the forward.
    # We keep NAFNet intact; K is only used in the loss branch (output-side consistency).
    # NewBPLayer is no longer wired into the network input here to avoid double crosstalk.

    logger.info(
        "[NewBP-Net] Created (Scenario B: no input-side K). kernel_type='%s', kernel_spec='%s', "
        "in_channels=%s, width=%s, enc_blks=%s, middle=%s, dec_blks=%s. Use CrosstalkPSF only in the loss branch.",
        kernel_type,
        kernel_spec,
        in_channels,
        nafnet_config.get('width'),
        nafnet_config.get('enc_blk_nums'),
        nafnet_config.get('middle_blk_num'),
        nafnet_config.get('dec_blk_nums'),
    )

    return base_model


def create_crosstalk_psf(psf_mode: str = 'mono', kernel_spec: str = 'P2'):
    """
    Helper to build a fixed PSF module for loss graph only.

    psf_mode: 'mono' -> single 3x3 kernel shared across channels; 'rgb' -> per-channel 3x3 kernels.
    kernel_spec: 'P2' for mono (panchromatic), 'B2' for rgb.
    """
    # map legacy names to new modes
    if psf_mode not in {'mono', 'rgb'}:
        raise ValueError("psf_mode must be 'mono' or 'rgb'")
    kernels = build_psf_kernels(psf_mode, kernel_spec)
    return CrosstalkPSF(mode=psf_mode, kernels=kernels)

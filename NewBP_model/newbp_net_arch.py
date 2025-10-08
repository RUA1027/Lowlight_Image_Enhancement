import os
import sys

import torch.nn as nn

NAFNET_ROOT = os.path.join(os.path.dirname(__file__), 'NAFNet')
if NAFNET_ROOT not in sys.path:
    sys.path.insert(0, NAFNET_ROOT)

from basicsr.models.archs.NAFNet_arch import NAFNet  # pyright: ignore[reportMissingImports]
from newbp_layer import NewBPLayer


def create_newbp_net(in_channels=3, kernel_type='panchromatic', kernel_spec='P2', nafnet_params=None):
    if nafnet_params is None:
        nafnet_params = {}

    if not isinstance(nafnet_params, dict):
        raise TypeError("nafnet_params must be a dictionary if provided.")

    nafnet_config = dict(nafnet_params)
    nafnet_config.setdefault('img_channel', in_channels)

    base_model = NAFNet(**nafnet_config)

    newbp_layer = NewBPLayer(
        in_channels=in_channels,
        kernel_type=kernel_type,
        kernel_spec=kernel_spec
    )

    original_intro = base_model.intro
    base_model.intro = nn.Sequential(newbp_layer, original_intro)

    print(
        f"[NewBP-Net] Created with kernel_type='{kernel_type}', kernel_spec='{kernel_spec}', "
        f"in_channels={in_channels}."
    )

    return base_model

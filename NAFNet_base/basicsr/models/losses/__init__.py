# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from copy import deepcopy

from basicsr.utils import get_root_logger
from basicsr.utils.registry import LOSS_REGISTRY

from .losses import CharbonnierLoss, L1Loss, MSELoss, PSNRLoss

try:
    from NewBP_model.losses import HybridLossPlus  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    HybridLossPlus = None


def build_loss(opt):
    """Create a loss instance from configuration dict."""
    if opt is None:
        raise ValueError('Loss config must not be None.')

    opt_copy = deepcopy(opt)
    loss_type = opt_copy.pop('type', None)
    if not loss_type:
        raise KeyError('Loss config must contain the key "type".')

    loss_cls = LOSS_REGISTRY.get(loss_type)
    if loss_cls is None:
        raise KeyError(f'Loss type {loss_type} is not registered in LOSS_REGISTRY.')

    loss = loss_cls(**opt_copy)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss


__all__ = [
    'L1Loss',
    'MSELoss',
    'PSNRLoss',
    'CharbonnierLoss',
    'build_loss',
]

if HybridLossPlus is not None:
    __all__.append('HybridLossPlus')

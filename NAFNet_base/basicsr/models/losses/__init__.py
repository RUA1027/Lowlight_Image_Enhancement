# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from .losses import (CharbonnierLoss, L1Loss, MSELoss, PSNRLoss)

try:
    from NewBP_model.losses import HybridLossPlus  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    HybridLossPlus = None

__all__ = [
    'L1Loss',
    'MSELoss',
    'PSNRLoss',
    'CharbonnierLoss',
]

if HybridLossPlus is not None:
    __all__.append('HybridLossPlus')

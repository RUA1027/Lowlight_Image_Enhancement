# ------------------------------------------------------------------------
# Compatibility shim to expose losses under `basicsr.losses` like upstream.
# ------------------------------------------------------------------------
from basicsr.models.losses import (  # noqa: F401
    CharbonnierLoss,
    L1Loss,
    MSELoss,
    PSNRLoss,
    build_loss,
)

__all__ = [
    'L1Loss',
    'MSELoss',
    'PSNRLoss',
    'CharbonnierLoss',
    'build_loss',
]

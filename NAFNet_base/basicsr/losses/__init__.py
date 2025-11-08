# ------------------------------------------------------------------------
# Minimal public API: expose build_loss from the local loss factory.
# ------------------------------------------------------------------------
from .losses import build_loss

__all__ = ['build_loss']

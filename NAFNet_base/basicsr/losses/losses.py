"""Expose build_loss from the canonical BasicSR models package."""

from basicsr.models.losses import build_loss as _build_loss


def build_loss(opt):
    return _build_loss(opt)

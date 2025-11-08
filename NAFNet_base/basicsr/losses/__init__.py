"""Thin shim: expose build_loss from basicsr.models.losses.

This avoids creating an extra indirection layer that can participate in
import cycles. The canonical implementation lives in
`basicsr.models.losses`.
"""

from basicsr.models.losses import build_loss

__all__ = ["build_loss"]

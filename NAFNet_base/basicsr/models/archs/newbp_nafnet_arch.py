"""Adapter to build the NewBP variant of NAFNet via project utilities."""

from __future__ import annotations

try:
    from NewBP_model.newbp_net_arch import create_newbp_net  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "NewBP_model package is required to instantiate the NewBP NAFNet variant."
    ) from exc


def NewBPNAFNet(**kwargs):
    return create_newbp_net(**kwargs)

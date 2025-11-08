"""Adapter to build the NewBP variant of NAFNet via project utilities."""

from __future__ import annotations

from typing import Any, Optional, Sequence

try:
    from basicsr.utils.registry import ARCH_REGISTRY
except ImportError:  # pragma: no cover
    from basicsr.utils import ARCH_REGISTRY  # type: ignore

try:
    from NewBP_model.newbp_net_arch import create_newbp_net as _create_newbp_net  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "NewBP_model package is required to instantiate the NewBP NAFNet variant."
    ) from exc


@ARCH_REGISTRY.register()
def create_newbp_net(
    in_channels: int = 3,
    kernel_type: str = "panchromatic",
    kernel_spec: str = "P2",
    width: Optional[int] = None,
    enc_blk_nums: Optional[Sequence[int]] = None,
    middle_blk_num: Optional[int] = None,
    dec_blk_nums: Optional[Sequence[int]] = None,
    nafnet_params: Optional[dict[str, Any]] = None,
    **nafnet_kwargs: Any,
):
    """Forward-compatible wrapper that preserves the BasicSR entrypoint signature."""

    return _create_newbp_net(
        in_channels=in_channels,
        kernel_type=kernel_type,
        kernel_spec=kernel_spec,
        width=width,
        enc_blk_nums=enc_blk_nums,
        middle_blk_num=middle_blk_num,
        dec_blk_nums=dec_blk_nums,
        nafnet_params=nafnet_params,
        **nafnet_kwargs,
    )


@ARCH_REGISTRY.register()
def NewBPNAFNet(**kwargs: Any):
    """Alias used by BasicSR network registry."""

    return create_newbp_net(**kwargs)

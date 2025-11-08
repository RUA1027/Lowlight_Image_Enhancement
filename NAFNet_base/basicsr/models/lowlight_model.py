"""Minimal LowlightModel compatible with BasicSR registries and SID datasets."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Dict

import torch
import torch.nn.functional as F

from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger

try:  # pragma: no cover - compatibility with legacy registry export
    from basicsr.utils.registry import ARCH_REGISTRY, MODEL_REGISTRY
except ImportError:  # pragma: no cover
    from basicsr.utils import ARCH_REGISTRY, MODEL_REGISTRY  # type: ignore

from basicsr.models.losses import build_loss


def build_network(opt: Dict) -> torch.nn.Module:
    """Instantiate a network via ARCH_REGISTRY to avoid direct basicsr.archs dependency."""

    if not isinstance(opt, dict):
        raise TypeError(f"build_network expects a dict, but received {type(opt)}")
    net_type = opt.get("type")
    if net_type is None:
        raise KeyError("network_g configuration missing required 'type' field.")

    net_cls = ARCH_REGISTRY.get(net_type)
    if net_cls is None:
        raise KeyError(
            f"ARCH_REGISTRY has no entry '{net_type}'. Ensure the corresponding arch module is imported."
        )
    kwargs = {k: v for k, v in opt.items() if k != "type"}
    return net_cls(**kwargs)


@MODEL_REGISTRY.register()
class LowlightModel(BaseModel):
    """Generic low-light enhancement wrapper supporting SID-style inputs."""

    def __init__(self, opt: Dict):
        super().__init__(opt)
        self.log_dict: Dict[str, float] = {}

        net_opt = deepcopy(opt["network_g"])
        self.net_g = build_network(net_opt)
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        load_path = opt["path"].get("pretrain_network_g")
        if load_path:
            strict = opt["path"].get("strict_load_g", True)
            param_key = opt["path"].get("param_key_g", "params")
            self.load_network(self.net_g, load_path, strict, param_key)

        self.cri_pix = None
        self.cri_perceptual = None
        self.cri_ssim = None

        if self.is_train:
            self.init_training_settings()
        else:
            self.net_g.eval()

    def init_training_settings(self) -> None:
        self.net_g.train()
        train_opt = self.opt.get("train", {})

        pixel_opt = deepcopy(train_opt.get("pixel_opt"))
        if pixel_opt:
            self.cri_pix = build_loss(pixel_opt).to(self.device)

        percep_opt = deepcopy(train_opt.get("perceptual_opt"))
        if percep_opt:
            self.cri_perceptual = build_loss(percep_opt).to(self.device)

        ssim_opt = deepcopy(train_opt.get("ssim_opt"))
        if ssim_opt:
            self.cri_ssim = build_loss(ssim_opt).to(self.device)

        if not any([self.cri_pix, self.cri_perceptual, self.cri_ssim]):
            get_root_logger().warning(
                "LowlightModel: no losses configured; optimize_parameters will fallback to L1."
            )

        optim_cfg = deepcopy(train_opt.get("optim_g", {"type": "AdamW", "lr": 1e-4}))
        optim_type = optim_cfg.pop("type", "AdamW").lower()
        params = [p for p in self.net_g.parameters() if p.requires_grad]
        if optim_type == "adamw":
            optimizer = torch.optim.AdamW(params, **optim_cfg)
        elif optim_type == "adam":
            optimizer = torch.optim.Adam(params, **optim_cfg)
        elif optim_type == "sgd":
            optimizer = torch.optim.SGD(params, **optim_cfg)
        else:
            raise NotImplementedError(f"LowlightModel does not support optimizer '{optim_type}'.")

        self.optimizer_g = optimizer
        self.optimizers.append(self.optimizer_g)
        if train_opt.get("scheduler"):
            self.setup_schedulers()

    def feed_data(self, data: Dict[str, torch.Tensor]) -> None:
        if "lq" in data and "gt" in data:
            self.lq = data["lq"].to(self.device)
            self.gt = data["gt"].to(self.device)
        elif "short" in data and "long" in data:
            self.lq = data["short"].to(self.device)
            self.gt = data["long"].to(self.device)
        else:
            raise ValueError(f"LowlightModel feed_data expects ('lq','gt') or ('short','long'), got {list(data.keys())}")

    def optimize_parameters(self) -> None:
        if not self.is_train:
            return

        self.optimizer_g.zero_grad()
        self.net_g.train()
        self.output = self.net_g(self.lq)

        total_loss = 0.0
        if self.cri_pix:
            total_loss = total_loss + self.cri_pix(self.output, self.gt)
        if self.cri_perceptual:
            total_loss = total_loss + self.cri_perceptual(self.output, self.gt)
        if self.cri_ssim:
            total_loss = total_loss + self.cri_ssim(self.output, self.gt)

        if (not torch.is_tensor(total_loss)) or total_loss.numel() == 0:
            total_loss = F.l1_loss(self.output, self.gt)

        total_loss.backward()
        self.optimizer_g.step()

        self.log_dict = {"loss_total": float(total_loss.detach().cpu())}

    @torch.no_grad()
    def test(self) -> None:
        self.net_g.eval()
        self.output = self.net_g(self.lq)
        if self.is_train:
            self.net_g.train()

    def get_current_visuals(self):
        visuals = OrderedDict()
        if hasattr(self, "lq"):
            visuals["lq"] = self.lq.detach().cpu()
        if hasattr(self, "gt"):
            visuals["gt"] = self.gt.detach().cpu()
        if hasattr(self, "output"):
            visuals["output"] = self.output.detach().cpu()
        return visuals

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img=False, rgb2bgr=True, use_image=True):
        self.net_g.eval()
        for data in dataloader:
            self.feed_data(data)
            with torch.no_grad():
                self.output = self.net_g(self.lq)
        self.net_g.train(not self.opt.get("val", {}).get("eval_only", False))

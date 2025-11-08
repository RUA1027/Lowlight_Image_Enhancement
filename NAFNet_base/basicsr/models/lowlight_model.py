"""Lightweight model wrapper for low-light enhancement experiments."""

from __future__ import annotations

import importlib
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY

loss_module = importlib.import_module("basicsr.models.losses")


def _build_loss(loss_opt: Optional[Dict[str, Any]], device: torch.device):
    if not loss_opt:
        return None
    cfg = deepcopy(loss_opt)
    loss_type = cfg.pop("type")
    cls = getattr(loss_module, loss_type)
    return cls(**cfg).to(device)


@MODEL_REGISTRY.register()
class LowlightModel(BaseModel):
    """Minimal training wrapper that works with SID-style datasets."""

    def __init__(self, opt: Dict[str, Any]):
        super().__init__(opt)
        self.log_dict: Dict[str, float] = {}

        self.net_g = define_network(deepcopy(opt["network_g"]))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        load_path = self.opt["path"].get("pretrain_network_g")
        if load_path:
            self.load_network(
                self.net_g,
                load_path,
                strict=self.opt["path"].get("strict_load_g", True),
                param_key=self.opt["path"].get("param_key", "params"),
            )

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

        self.cri_pix = _build_loss(deepcopy(train_opt.get("pixel_opt")), self.device)
        self.cri_perceptual = _build_loss(deepcopy(train_opt.get("perceptual_opt")), self.device)
        self.cri_ssim = _build_loss(deepcopy(train_opt.get("ssim_opt")), self.device)
        if not any([self.cri_pix, self.cri_perceptual, self.cri_ssim]):
            get_root_logger().warning("All configured losses missing; falling back to L1 inside optimize_parameters.")

        optim_cfg = deepcopy(train_opt.get("optim_g", {}))
        optim_type = optim_cfg.pop("type", "Adam")
        params = [p for p in self.net_g.parameters() if p.requires_grad]
        if optim_type == "Adam":
            optimizer = torch.optim.Adam(params, **optim_cfg)
        elif optim_type == "AdamW":
            optimizer = torch.optim.AdamW(params, **optim_cfg)
        elif optim_type == "SGD":
            optimizer = torch.optim.SGD(params, **optim_cfg)
        else:
            raise ValueError(f"Unsupported optimizer type: {optim_type}")
        self.optimizer_g = optimizer
        self.optimizers.append(self.optimizer_g)

        self.setup_schedulers()

    def feed_data(self, data: Dict[str, torch.Tensor]) -> None:
        if "lq" in data and "gt" in data:
            self.lq = data["lq"].to(self.device)
            self.gt = data["gt"].to(self.device)
        elif "short" in data and "long" in data:
            self.lq = data["short"].to(self.device)
            self.gt = data["long"].to(self.device)
        else:
            raise ValueError("LowlightModel.feed_data expects ('lq','gt') or ('short','long') keys.")

        self.meta = {}
        if "expo_ratio" in data:
            self.meta["expo_ratio"] = data["expo_ratio"].to(self.device)
        if "ratio" in data:
            self.meta["ratio"] = data["ratio"]
        if "metadata" in data:
            self.meta["metadata"] = data["metadata"]

    def optimize_parameters(self) -> None:
        if not self.is_train:
            raise RuntimeError("optimize_parameters can only be called during training mode.")

        for opt in self.optimizers:
            opt.zero_grad()

        self.output = self.net_g(self.lq)
        total_loss = 0.0
        loss_terms = {}

        if self.cri_pix:
            loss_pix = self.cri_pix(self.output, self.gt)
            total_loss = total_loss + loss_pix
            loss_terms["loss_pix"] = float(loss_pix.detach().cpu())

        if self.cri_perceptual:
            loss_perc = self.cri_perceptual(self.output, self.gt)
            total_loss = total_loss + loss_perc
            loss_terms["loss_perc"] = float(loss_perc.detach().cpu())

        if self.cri_ssim:
            loss_ssim = self.cri_ssim(self.output, self.gt)
            total_loss = total_loss + loss_ssim
            loss_terms["loss_ssim"] = float(loss_ssim.detach().cpu())

        if isinstance(total_loss, float) and total_loss == 0.0:
            fallback = F.l1_loss(self.output, self.gt)
            total_loss = fallback
            loss_terms["loss_l1_fallback"] = float(fallback.detach().cpu())

        total_loss.backward()
        for opt in self.optimizers:
            opt.step()

        self.log_dict = {"loss": float(total_loss.detach().cpu())}
        self.log_dict.update(loss_terms)

    def test(self) -> None:
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
        if self.is_train:
            self.net_g.train()

    def get_current_visuals(self) -> Dict[str, torch.Tensor]:
        visuals = {}
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

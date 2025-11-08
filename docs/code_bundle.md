# Code Bundle Snapshot

Below are the requested files with their full contents, in order. If a file was not found, it is noted as “没有找到”.

## basicsr/__init__.py

Path: `basicsr/__init__.py`

```
"""Proxy package to ensure `import basicsr` resolves to NAFNet_base/basicsr."""

from __future__ import annotations

import pathlib

_PKG_DIR = pathlib.Path(__file__).resolve().parent
_REAL_PKG_DIR = _PKG_DIR.parent / "NAFNet_base" / "basicsr"
_REAL_INIT = _REAL_PKG_DIR / "__init__.py"

if not _REAL_PKG_DIR.exists():
    raise ImportError(f"Expected basicsr sources at {_REAL_PKG_DIR}, but the directory is missing.")

__file__ = str(_REAL_INIT)
__path__ = [str(_REAL_PKG_DIR)]

if __spec__ is not None:  # pragma: no cover
    __spec__.origin = __file__
    __spec__.submodule_search_locations = __path__

code = compile(_REAL_INIT.read_text(encoding="utf-8"), __file__, "exec")
exec(code, globals(), globals())
```

## NAFNet_base/basicsr/__init__.py

Path: `NAFNet_base/basicsr/__init__.py`

```

```

## NAFNet_base/basicsr/models/__init__.py

Path: `NAFNet_base/basicsr/models/__init__.py`

```
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
from os import path as osp

from basicsr.utils import get_root_logger, scandir

try:  # pragma: no cover - registry may be re-exported elsewhere
    from basicsr.utils.registry import MODEL_REGISTRY
except ImportError:  # pragma: no cover
    from basicsr.utils import MODEL_REGISTRY  # type: ignore

# Collect available model module names (without importing them immediately).
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(model_folder)
    if v.endswith('_model.py')
]

_MODEL_CACHE = {}

from . import lowlight_model  # noqa: F401 - ensure LowlightModel registers itself


def _iterate_model_modules():
    """Yield imported model modules, importing lazily to avoid heavy deps at import time."""
    for file_name in model_filenames:
        if file_name not in _MODEL_CACHE:
            _MODEL_CACHE[file_name] = importlib.import_module(f'basicsr.models.{file_name}')
        yield _MODEL_CACHE[file_name]


def create_model(opt):
    """Create model.

    Args:
        opt (dict): Configuration. It constains:
            model_type (str): Model type.
    """
    model_type = opt['model_type']

    model_cls = MODEL_REGISTRY.get(model_type) if MODEL_REGISTRY is not None else None

    if model_cls is None:
        # Fallback to scanning modules until the requested model shows up.
        for module in _iterate_model_modules():
            model_cls = getattr(module, model_type, None)
            if model_cls is not None:
                break

    if model_cls is None and MODEL_REGISTRY is not None:
        model_cls = MODEL_REGISTRY.get(model_type)

    if model_cls is None:
        raise ValueError(f'Model {model_type} is not found.')

    model = model_cls(opt)

    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model
```

## NAFNet_base/basicsr/models/lowlight_model.py

Path: `NAFNet_base/basicsr/models/lowlight_model.py`

```
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

try:  # Prefer the canonical basicsr.losses package when available
    from basicsr.losses import build_loss
except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover - fallback for local repo
    from basicsr.models.losses import build_loss  # type: ignore


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
```

## NAFNet_base/basicsr/models/losses/__init__.py

Path: `NAFNet_base/basicsr/models/losses/__init__.py`

```
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
```

## NAFNet_base/basicsr/models/losses/losses.py

Path: `NAFNet_base/basicsr/models/losses/losses.py`

```
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss
from basicsr.utils.registry import LOSS_REGISTRY

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    diff = pred - target
    return torch.sqrt(diff * diff + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Robust Charbonnier loss (smooth L1)."""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(
                f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * charbonnier_loss(
            pred, target, weight, reduction=self.reduction, eps=self.eps)

@LOSS_REGISTRY.register()
class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
```

## NAFNet_base/basicsr/losses/__init__.py

Path: `NAFNet_base/basicsr/losses/__init__.py`

```
# ------------------------------------------------------------------------
# Minimal public API: expose build_loss from the local loss factory.
# ------------------------------------------------------------------------
from .losses import build_loss

__all__ = ['build_loss']
```

## NAFNet_base/basicsr/losses/losses.py（如果有）

Path: `NAFNet_base/basicsr/losses/losses.py`

```
"""Expose build_loss from the canonical BasicSR models package."""

from basicsr.models.losses import build_loss as _build_loss


def build_loss(opt):
    return _build_loss(opt)
```

## basicsr/train.py

Path: `basicsr/train.py`

没有找到

## NAFNet_base/basicsr/train.py（如果有）

Path: `NAFNet_base/basicsr/train.py`

```
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import argparse
import datetime
import logging
import math
import random
import time
import torch
from os import path as osp

import sys
import os
# Ensure Python can import the local 'basicsr' package regardless of where the script is launched from.
# 1) Add the parent directory (NAFNet_base) so that 'import basicsr' resolves to this folder.
_here = os.path.abspath(os.path.dirname(__file__))
_pkg_root = os.path.abspath(os.path.join(_here, '..'))  # NAFNet_base
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
# 2) Optionally add the repository root to support other relative imports if needed.
_repo_root = os.path.abspath(os.path.join(_pkg_root, '..'))  # project root
if _repo_root not in sys.path:
    sys.path.append(_repo_root)
from basicsr.utils import MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str, init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, set_random_seed
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse
from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--input_path', type=str, required=False, help='The path to the input image. For single image inference only.')
    parser.add_argument('--output_path', type=str, required=False, help='The path to the output image. For single image inference only.')

    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # number of GPUs visible to this process (used by dataloader construction)
    # In non-distributed mode, use all visible GPUs; in distributed, one GPU per process.
    if opt['dist']:
        # In distributed mode each process handles exactly one GPU; if CUDA not available, fall back to CPU with 0.
        if torch.cuda.is_available():
            opt['num_gpu'] = 1
        else:
            opt['num_gpu'] = 0
    else:
        # Non-distributed: reflect actual visible GPU count; allow 0 for pure CPU so downstream device selection uses CPU.
        try:
            visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            visible = 0
        opt['num_gpu'] = int(visible)
        if opt['num_gpu'] == 0:
            print('CUDA not available: running in pure CPU mode (num_gpu=0).', flush=True)

```

```
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):
    epoch = start_epoch
    while current_iter <= total_iters:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data, is_val=False)
            result_code = model.optimize_parameters(current_iter, tb_logger)
            # if result_code == -1 and tb_logger:
            #     print('loss explode .. ')
            #     exit(0)
            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                # print('msg logger .. ', current_iter)
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0 or current_iter == 1000):
            # if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
                use_image = opt['val'].get('use_image', True)
                model.validation(val_loader, current_iter, tb_logger,
                                 opt['val']['save_img'], rgb2bgr, use_image )
                log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)


            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter
        epoch += 1

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        rgb2bgr = opt['val'].get('rgb2bgr', True)
        use_image = opt['val'].get('use_image', True)
        metric = model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'], rgb2bgr, use_image)
        # if tb_logger:
        #     print('xxresult! ', opt['name'], ' ', metric)
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    import os
    os.environ['GRPC_POLL_STRATEGY']='epoll1'
    main()
```

## NAFNet_base/configs/colab/sid_newbp_mono_colab.yml

Path: `NAFNet_base/configs/colab/sid_newbp_mono_colab.yml`

没有找到


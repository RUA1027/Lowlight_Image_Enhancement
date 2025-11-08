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

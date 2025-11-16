# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import os
import yaml
from collections import OrderedDict
from os import path as osp
from pathlib import Path

from basicsr.utils.sid_paths import expand_with_sid_root


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def _expand_path(path_value):
    """Expand environment variables and user markers, normalising separators."""

    if path_value is None:
        return None

    text = str(path_value).strip()
    if not text:
        return text

    expanded = os.path.expandvars(text)
    # Normalise Windows backslashes to forward slashes before pathlib handling
    expanded = expanded.replace('\\', '/')
    path = Path(expanded).expanduser()
    # Keep relative paths relative while ensuring consistent separators
    return path.as_posix()


def _normalise_dataset_paths(dataset_opt):
    manifest_path = dataset_opt.get('manifest_path')
    if manifest_path:
        resolved = expand_with_sid_root(manifest_path)
        if resolved:
            dataset_opt['manifest_path'] = resolved.as_posix()

    io_backend = dataset_opt.get('io_backend')
    if isinstance(io_backend, dict):
        db_paths = io_backend.get('db_paths')
        if isinstance(db_paths, list):
            normalised = []
            for path_value in db_paths:
                resolved = expand_with_sid_root(path_value)
                normalised.append(resolved.as_posix() if resolved else path_value)
            io_backend['db_paths'] = normalised
        path_dict = io_backend.get('paths')
        if isinstance(path_dict, dict):
            for key, value in list(path_dict.items()):
                resolved = expand_with_sid_root(value)
                if resolved:
                    path_dict[key] = resolved.as_posix()

    for legacy_key in ('short_lmdb', 'long_lmdb'):
        value = dataset_opt.get(legacy_key)
        if value:
            resolved = expand_with_sid_root(value)
            if resolved:
                dataset_opt[legacy_key] = resolved.as_posix()


def parse(opt_path, is_train=True):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    opt['is_train'] = is_train

    # datasets
    if 'datasets' in opt:
        for phase, dataset in opt['datasets'].items():
            # for several datasets, e.g., test_1, test_2
            phase = phase.split('_')[0]
            dataset['phase'] = phase
            if 'scale' in opt:
                dataset['scale'] = opt['scale']
            if dataset.get('dataroot_gt') is not None:
                dataset['dataroot_gt'] = _expand_path(dataset['dataroot_gt'])
            if dataset.get('dataroot_lq') is not None:
                dataset['dataroot_lq'] = _expand_path(dataset['dataroot_lq'])
            _normalise_dataset_paths(dataset)

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key
                                  or 'pretrain_network' in key):
            opt['path'][key] = _expand_path(val)
    opt['path']['root'] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments',
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg

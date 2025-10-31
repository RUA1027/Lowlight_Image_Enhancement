"""
End-to-end data/config/training smoke tests.

Goals:
- Parse project YAML configs via basicsr utils.
- Instantiate dataset/dataloader using the debug manifest/LMDB placeholders.
- Execute a short training loop (few iterations) through basicsr/train.py utilities.

These tests rely on the synthetic assets in data/debug_sid; adjust paths if using real data.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NAFNET_BASE = PROJECT_ROOT / "NAFNet_base"
if str(NAFNET_BASE) not in sys.path:
    sys.path.insert(0, str(NAFNET_BASE))

from basicsr.data import create_dataloader, create_dataset  # type: ignore
from basicsr.models import create_model  # type: ignore
from basicsr.utils.options import parse  # type: ignore
from basicsr.utils import get_time_str  # type: ignore


def _load_yaml(path: Path) -> dict:
    opt = parse(str(path), is_train=True)
    assert isinstance(opt, dict)
    return opt


def _patch_debug_paths(opt: dict, root: Path) -> dict:
    manifest = root / "manifest_sid_debug.json"
    short_png = root / "short"
    long_png = root / "long"
    short_lmdb = root / "train_short_debug.lmdb"
    long_lmdb = root / "train_long_debug.lmdb"

    if not short_lmdb.exists() or not long_lmdb.exists():
        pytest.skip(
            f"缺少调试用 LMDB：{short_lmdb} 或 {long_lmdb}。请使用 tools/create_sid_lmdb.py 生成小型 LMDB 后再运行。"
        )

    for phase in ("train", "val"):
        ds_opt = opt["datasets"][phase]
        ds_opt["manifest_path"] = manifest.as_posix()
        ds_opt["io_backend"]["db_paths"] = [short_lmdb.as_posix(), long_lmdb.as_posix()]

    opt["path"]["experiments_root"] = str(PROJECT_ROOT / "logs" / f"pytest_smoke_{get_time_str()}")
    opt["path"]["models"] = os.path.join(opt["path"]["experiments_root"], "models")
    opt["path"]["training_states"] = os.path.join(opt["path"]["experiments_root"], "states")
    opt["path"]["log"] = os.path.join(opt["path"]["experiments_root"], "log")
    opt["path"]["visualization"] = os.path.join(opt["path"]["experiments_root"], "viz")
    opt["path"]["root"] = str(PROJECT_ROOT)
    opt["name"] = "pytest_smoke_debug"
    opt["is_train"] = True
    return opt


@pytest.fixture(scope="module")
def debug_manifest_root() -> Path:
    root = PROJECT_ROOT / "data" / "debug_sid"
    if not root.exists():
        pytest.skip("Debug SID data not found; place assets under data/debug_sid to enable pipeline tests.")
    return root


def test_dataset_loader_debug_assets(debug_manifest_root: Path) -> None:
    opt = _load_yaml(PROJECT_ROOT / "configs" / "debug" / "sid_newbp_mono_debug.yml")
    opt = _patch_debug_paths(opt, debug_manifest_root)

    train_set = create_dataset(opt["datasets"]["train"])
    train_loader = create_dataloader(train_set, opt["datasets"]["train"], num_gpu=1, dist=False, sampler=None, seed=42)

    batch = next(iter(train_loader))
    assert isinstance(batch, dict)
    assert "lq" in batch and "gt" in batch
    assert batch["lq"].ndim == 4 and batch["gt"].ndim == 4


@pytest.mark.timeout(120)
def test_training_entrypoint_smoke(debug_manifest_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opt = _load_yaml(PROJECT_ROOT / "configs" / "debug" / "sid_newbp_mono_debug.yml")
    opt = _patch_debug_paths(opt, debug_manifest_root)
    opt["train"]["total_iter"] = 5
    opt["logger"]["use_tb_logger"] = False

    if opt["datasets"]["train"]["num_worker_per_gpu"] > 0:
        opt["datasets"]["train"]["num_worker_per_gpu"] = 0
        opt["datasets"]["val"]["num_worker_per_gpu"] = 0

    model = create_model(opt)
    train_set = create_dataset(opt["datasets"]["train"])
    train_loader = create_dataloader(train_set, opt["datasets"]["train"], num_gpu=1, dist=False, sampler=None, seed=42)

    current_iter = 0
    for data in train_loader:
        current_iter += 1
        model.feed_data(data)
        model.optimize_parameters(current_iter)
        log_dict = model.get_current_log()
        assert all(key in log_dict for key in ("l1", "Perc")) or log_dict, "Missing expected log keys."
        if current_iter >= opt["train"]["total_iter"]:
            break

    assert current_iter >= opt["train"]["total_iter"]

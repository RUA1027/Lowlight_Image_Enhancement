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
import json


def _load_yaml(path: Path) -> dict:
    opt = parse(str(path), is_train=True)
    assert isinstance(opt, dict)
    return opt


def _discover_subset_names(manifest_path: Path) -> set[str]:
    if not manifest_path.is_file():
        return set()
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(entries, list):
            return {str(e.get("subset", "")) for e in entries if isinstance(e, dict) and e.get("subset")}
    except Exception:
        pass
    return set()


def _choose_subset(all_subsets: set[str], *preferences: str) -> str | None:
    if not all_subsets:
        return None
    # direct preference exact match first
    for pref in preferences:
        if pref in all_subsets:
            return pref
    # fuzzy: startswith any preference token
    for pref in preferences:
        for s in sorted(all_subsets):
            if s.startswith(pref):
                return s
    # fallback: any subset containing token
    for pref in preferences:
        for s in sorted(all_subsets):
            if pref in s:
                return s
    # ultimate fallback: first alphabetical
    return sorted(all_subsets)[0]


def _lmdb_pair(root: Path, subset: str) -> tuple[Path, Path]:
    return root / f"{subset}_short.lmdb", root / f"{subset}_long.lmdb"


def _patch_debug_paths(opt: dict, root: Path) -> dict:
    """Patch dataset options to point to discovered manifest/LMDB assets.

    Resolution order:
    1. Parse manifest to discover subset names (e.g. train_small, val_small).
    2. Prefer subsets starting with/containing 'train' for training phase; 'val' for validation.
    3. Fall back to legacy fixed names train_short_debug.lmdb / train_long_debug.lmdb if subset LMDB absent.
    4. Skip test gracefully if neither subset-based nor legacy LMDBs exist.
    """

    manifest = root / "manifest_sid_debug.json"
    all_subsets = _discover_subset_names(manifest)

    # choose subset names
    train_subset = _choose_subset(all_subsets, "train", "debug", "small")
    val_subset = _choose_subset(all_subsets - {train_subset} if train_subset else all_subsets, "val", "small")

    # subset-based LMDB paths
    subset_train_short, subset_train_long = _lmdb_pair(root, train_subset) if train_subset else (None, None)
    subset_val_short, subset_val_long = _lmdb_pair(root, val_subset) if val_subset else (None, None)

    # legacy fallback paths
    legacy_short = root / "train_short_debug.lmdb"
    legacy_long = root / "train_long_debug.lmdb"

    def _ensure_pair(short_p: Path | None, long_p: Path | None) -> tuple[Path, Path] | None:
        if short_p is None or long_p is None:
            return None
        if short_p.exists() and long_p.exists():
            return short_p, long_p
        return None

    train_pair = _ensure_pair(subset_train_short, subset_train_long) or _ensure_pair(legacy_short, legacy_long)
    val_pair = _ensure_pair(subset_val_short, subset_val_long) or train_pair  # allow sharing if val subset absent

    if train_pair is None:
        pytest.skip(
            "未找到训练 LMDB。请运行 NAFNet_base/tools/create_sid_lmdb.py 生成，或确保命名 train_*_short.lmdb / train_short_debug.lmdb 存在。"
        )

    if val_pair is None:
        pytest.skip(
            "未找到验证 LMDB。请生成对应 val_*_short.lmdb 或使用与训练相同的 subset。"
        )

    (train_short_lmdb, train_long_lmdb) = train_pair
    (val_short_lmdb, val_long_lmdb) = val_pair

    for phase, (short_lmdb, long_lmdb) in {
        "train": (train_short_lmdb, train_long_lmdb),
        "val": (val_short_lmdb, val_long_lmdb),
    }.items():
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
    loader_iter = iter(train_loader)
    # Force small deterministic length >= total_iter by cycling if needed
    loader_iter = iter(train_loader)

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

    # Tiny debug assets (64x64) => constrain crop/patch to fit
    opt["datasets"]["train"]["patch_size"] = 64
    opt["datasets"]["train"]["random_crop"] = False

    # Provide minimal runtime defaults to satisfy BaseModel expectations when running CPU-only smoke test.
    if "num_gpu" not in opt:
        opt["num_gpu"] = 0
    if "dist" not in opt:
        opt["dist"] = False
    if "rank" not in opt:
        opt["rank"] = 0
    if "world_size" not in opt:
        opt["world_size"] = 1

    if opt["datasets"]["train"]["num_worker_per_gpu"] > 0:
        opt["datasets"]["train"]["num_worker_per_gpu"] = 0
        opt["datasets"]["val"]["num_worker_per_gpu"] = 0
    # Ensure persistent_workers disabled when num workers are zero (PyTorch constraint)
    for phase in ("train", "val"):
        if opt["datasets"][phase].get("num_worker_per_gpu", 0) == 0:
            opt["datasets"][phase]["persistent_workers"] = False

    model = create_model(opt)
    train_set = create_dataset(opt["datasets"]["train"])
    train_loader = create_dataloader(train_set, opt["datasets"]["train"], num_gpu=1, dist=False, sampler=None, seed=42)
    loader_iter = iter(train_loader)

    current_iter = 0
    while current_iter < opt["train"]["total_iter"]:
        try:
            data = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            data = next(loader_iter)
        current_iter += 1
        model.feed_data(data)
        model.optimize_parameters(current_iter, tb_logger=None)
        log_dict = model.get_current_log()
        assert isinstance(log_dict, dict)

    assert current_iter >= opt["train"]["total_iter"]

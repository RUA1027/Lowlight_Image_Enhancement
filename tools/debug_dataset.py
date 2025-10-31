#!/usr/bin/env python
"""
Dataset sanity checker for the SID pipeline.

Usage examples (CPU-friendly):
    python tools/debug_dataset.py --manifest data/debug_sid/manifest_sid_debug.json \
        --short-root data/debug_sid/short --long-root data/debug_sid/long --limit 2

    python tools/debug_dataset.py --manifest /path/to/manifest_sid.json \
        --short-root /content/drive/.../SID_raw/Sony/short \
        --long-root /content/drive/.../SID_raw/Sony/long --limit 1 --inspect
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError("Pillow 未安装，请先 pip install Pillow") from exc

try:
    import rawpy  # type: ignore
except Exception:
    rawpy = None

try:
    import lmdb
except Exception:
    lmdb = None


def load_manifest(manifest_path: Path) -> list[dict]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("manifest 文件必须是包含字典的列表")
    return data


def check_ratio(short_exposure: float, long_exposure: float, ratio: float, tol: float = 1e-3) -> bool:
    if short_exposure <= 0 or long_exposure <= 0:
        return False
    expected = long_exposure / short_exposure
    return math.isclose(expected, ratio, rel_tol=tol, abs_tol=tol)


def try_open_image(path: Path, inspect: bool) -> tuple[int, int, int]:
    suffix = path.suffix.lower()
    if suffix in {".arw", ".dng", ".nef", ".cr2"}:
        if rawpy is None:
            raise RuntimeError(f"检测到 RAW 文件 {path.name}，但 rawpy 未安装，无法读取")
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True)
            if inspect:
                print(f"    RAW -> RGB shape: {rgb.shape}, dtype={rgb.dtype}")
            return rgb.shape[1], rgb.shape[0], rgb.shape[2]

    with Image.open(path) as img:
        arr = np.array(img)
        if inspect:
            print(f"    读取图像: mode={img.mode}, size={img.size}, dtype={arr.dtype}")
        h, w = arr.shape[:2]
        c = 1 if arr.ndim == 2 else arr.shape[2]
        return w, h, c


def check_lmdb_entry(env: lmdb.Environment, key: str, inspect: bool) -> bool:
    if env is None:
        raise RuntimeError("未安装 lmdb，无法检查 LMDB 数据库")
    with env.begin(write=False) as txn:
        value = txn.get(key.encode("utf-8"))
        if value is None:
            return False
        if inspect:
            print(f"    LMDB key={key}，字节长度={len(value)}")
    return True


def iter_filtered(manifest: list[dict], subset: Optional[str]) -> Iterable[dict]:
    if subset:
        return (entry for entry in manifest if entry.get("subset") == subset)
    return iter(manifest)


def sanity_check(
    manifest_path: Path,
    short_root: Optional[Path],
    long_root: Optional[Path],
    lmdb_short: Optional[Path],
    lmdb_long: Optional[Path],
    subset: Optional[str],
    limit: int,
    inspect: bool,
) -> None:
    manifest = load_manifest(manifest_path)
    total = len(manifest)
    print(f"Manifest: {manifest_path} ，共 {total} 条记录")

    if total == 0:
        raise RuntimeError("manifest 文件为空")

    if short_root and not short_root.is_dir():
        raise FileNotFoundError(f"短曝光目录不存在: {short_root}")
    if long_root and not long_root.is_dir():
        raise FileNotFoundError(f"长曝光目录不存在: {long_root}")

    env_short = lmdb.open(str(lmdb_short), readonly=True, lock=False, readahead=False) if lmdb_short else None
    env_long = lmdb.open(str(lmdb_long), readonly=True, lock=False, readahead=False) if lmdb_long else None

    try:
        checked = 0
        for entry in iter_filtered(manifest, subset):
            pair_id = entry.get("pair_id", "<unknown>")
            short_key = entry.get("short_key")
            long_key = entry.get("long_key")
            short_exp = float(entry.get("short_exposure", 0))
            long_exp = float(entry.get("long_exposure", 0))
            ratio = float(entry.get("exposure_ratio", 0))

            print(f"检查 pair: {pair_id} (subset={entry.get('subset', 'unknown')})")

            if not check_ratio(short_exp, long_exp, ratio):
                raise RuntimeError(
                    f"曝光比不一致: long/short={long_exp/short_exp:.6f}, manifest ratio={ratio:.6f}"
                )

            if short_root:
                short_path = short_root / short_key
                if not short_path.is_file():
                    raise FileNotFoundError(f"短曝光文件缺失: {short_path}")
                w, h, c = try_open_image(short_path, inspect)
                print(f"    短曝光图像存在，分辨率 {w}x{h}，通道 {c}")

            if long_root:
                long_path = long_root / long_key
                if not long_path.is_file():
                    raise FileNotFoundError(f"长曝光文件缺失: {long_path}")
                w, h, c = try_open_image(long_path, inspect)
                print(f"    长曝光图像存在，分辨率 {w}x{h}，通道 {c}")

            if env_short:
                if not check_lmdb_entry(env_short, short_key, inspect):
                    raise RuntimeError(f"LMDB 中缺失短曝光 key: {short_key}")
                print("    LMDB 短曝光 key 存在")

            if env_long:
                if not check_lmdb_entry(env_long, long_key, inspect):
                    raise RuntimeError(f"LMDB 中缺失长曝光 key: {long_key}")
                print("    LMDB 长曝光 key 存在")

            checked += 1
            if 0 < limit == checked:
                break

        if checked == 0:
            raise RuntimeError(f"指定 subset={subset} 时未匹配到任何记录")

        print(f"校验完成，共检查 {checked} 条记录，无错误。")
    finally:
        if env_short:
            env_short.close()
        if env_long:
            env_long.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SID 数据集完整性检查脚本")
    parser.add_argument("--manifest", type=Path, required=True, help="manifest_sid.json 路径")
    parser.add_argument("--short-root", type=Path, help="短曝光 PNG/RAW 目录")
    parser.add_argument("--long-root", type=Path, help="长曝光 PNG/RAW 目录")
    parser.add_argument("--lmdb-short", type=Path, help="短曝光 LMDB 目录")
    parser.add_argument("--lmdb-long", type=Path, help="长曝光 LMDB 目录")
    parser.add_argument("--subset", type=str, help="仅检查指定 subset (train/val/test 等)")
    parser.add_argument("--limit", type=int, default=3, help="最多检查多少条记录 (0 表示全部)")
    parser.add_argument("--inspect", action="store_true", help="打印图像形状、LMDB 字节长度等详细信息")
    args = parser.parse_args()

    sanity_check(
        manifest_path=args.manifest,
        short_root=args.short_root,
        long_root=args.long_root,
        lmdb_short=args.lmdb_short,
        lmdb_long=args.lmdb_long,
        subset=args.subset,
        limit=args.limit,
        inspect=args.inspect,
    )


if __name__ == "__main__":
    main()

"""Dataset wrapper for the Sony SID dataset with LMDB-based storage.

This dataset converts the original RAW `.ARW` files to 16-bit PNG, stores them in
LMDB for efficient I/O on Colab, and exposes aligned short/long exposure pairs
to the BasicSR training pipeline. Additional tensors (e.g. original short RAW,
exposure ratio) are returned so that custom physics-aware losses can be evaluated.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils import data as torch_data

from basicsr.data.transforms import paired_random_crop
from basicsr.utils import FileClient, img2tensor
from basicsr.utils.sid_paths import expand_with_sid_root

MAX_16BIT_VALUE = 65535.0


def _expand_to_path(path_value: Union[str, os.PathLike[str]]) -> Path:
    """Return a pathlib.Path with env vars expanded and SID_ROOT awareness."""

    path = expand_with_sid_root(path_value)
    if path is None:
        raise ValueError("Received an empty path while resolving dataset inputs.")
    return path


def _load_png_uint16(buffer: bytes) -> np.ndarray:
    """Decode a PNG buffer (byte string) to uint16 numpy array in RGB order."""
    if buffer is None:
        raise ValueError("Received empty buffer when decoding PNG.")

    np_buffer = np.frombuffer(buffer, np.uint8)
    img = cv2.imdecode(np_buffer, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to decode PNG buffer into an image.")
    # 允许占位图为 uint8（调试用途），自动提升到 uint16 以兼容管线
    if img.dtype == np.uint8:
        img = (img.astype(np.uint16) * 257)
    if img.dtype != np.uint16:
        raise TypeError(f"Expected uint16 image, got dtype={img.dtype}.")
    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image, got shape {img.shape}.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class SonySIDLMDBDataset(torch_data.Dataset):
    """Sony SID dataset backed by LMDB (or disk) for BasicSR training."""

    def __init__(self, opt: dict):
        super().__init__()

        self.opt = opt
        self.phase: str = opt.get("phase", "train")
        self.patch_size: Optional[int] = opt.get("patch_size")
        self.samples_per_pair: int = int(opt.get("samples_per_pair", 1))
        self.random_crop: bool = opt.get("random_crop", True)
        self.return_metadata: bool = opt.get("return_metadata", False)
        self.rng = np.random.default_rng(opt.get("seed", None))

        # Support environment variables in manifest path
        manifest_path = _expand_to_path(opt["manifest_path"])
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        subset = opt.get("subset", self.phase)
        self.subset = subset
        allowed_ids = set(opt.get("allowed_pair_ids", []))
        entries: list[dict] = []

        for record in manifest_data:
            if record.get("subset") != subset:
                continue
            if allowed_ids and record["pair_id"] not in allowed_ids:
                continue
            entries.append(record)

        if not entries:
            raise RuntimeError(f"No entries available for subset='{subset}'.")

        self.entries = entries
        self._num_pairs = len(entries)

        io_backend_opt = dict(opt.get("io_backend", {}))
        # Fallback: allow direct short/long lmdb paths in opt for legacy configs
        if not io_backend_opt and ("short_lmdb" in opt or "long_lmdb" in opt):
            io_backend_opt = {
                "type": "lmdb",
                "db_paths": [opt.get("short_lmdb"), opt.get("long_lmdb")],
                "client_keys": ["short", "long"],
            }
        if not io_backend_opt:
            raise KeyError("Dataset option must provide 'io_backend' or 'short_lmdb'/'long_lmdb' paths.")
        self.file_client: Optional[FileClient] = None
        backend_type = io_backend_opt.pop("type")
        self.io_backend_type = backend_type

        if backend_type == "lmdb":
            db_paths = io_backend_opt.get("db_paths")
            client_keys = io_backend_opt.get("client_keys")
            if not db_paths or not client_keys:
                raise KeyError("LMDB backend requires 'db_paths' and 'client_keys'.")
            # Expand environment variables / ~ for each path entry
            db_paths = [_expand_to_path(p) for p in db_paths]
            if any(not path for path in db_paths):
                raise ValueError(f"Invalid LMDB paths provided: {db_paths}")
            self.file_client = FileClient(
                backend="lmdb",
                db_paths=[str(p) for p in db_paths],
                client_keys=client_keys,
                readonly=True,
                lock=False,
                readahead=False,
            )
        elif backend_type == "disk":
            short_path = _expand_to_path(io_backend_opt["paths"]["short"])
            long_path = _expand_to_path(io_backend_opt["paths"]["long"])
            self.root_short = short_path
            self.root_long = long_path
            if not self.root_short.is_dir() or not self.root_long.is_dir():
                raise FileNotFoundError(
                    f"Disk backend paths must exist. short={self.root_short}, long={self.root_long}"
                )
        else:
            raise ValueError(f"Unsupported io_backend type: {backend_type}")

    def __len__(self) -> int:
        return self._num_pairs * self.samples_per_pair

    def _resolve_disk_path(self, key: str, is_short: bool) -> Path:
        root = self.root_short if is_short else self.root_long  # type: ignore[attr-defined]
        return root / key

    def _fetch_png(self, key: str, *, client_key: str) -> np.ndarray:
        if self.io_backend_type == "lmdb":
            assert self.file_client is not None
            buffer = self.file_client.get(key, client_key=client_key)
            return _load_png_uint16(buffer)

        # disk backend
        path = self._resolve_disk_path(key, is_short=(client_key == "short"))
        with path.open("rb") as f:
            buffer = f.read()
        return _load_png_uint16(buffer)

    def _maybe_random_crop(
        self,
        short_img: np.ndarray,
        long_img: np.ndarray,
        short_raw: np.ndarray,
        long_raw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.patch_size is None or self.phase != "train":
            return short_img, long_img, short_raw, long_raw

        h, w, _ = short_img.shape
        ps = self.patch_size
        if ps > h or ps > w:
            raise ValueError(f"Patch size {ps} exceeds source dimensions {(h, w)}.")

        if self.random_crop:
            top = self.rng.integers(0, h - ps + 1)
            left = self.rng.integers(0, w - ps + 1)
        else:
            top = (h - ps) // 2
            left = (w - ps) // 2

        bottom = top + ps
        right = left + ps

        def _crop(arr: np.ndarray) -> np.ndarray:
            return arr[top:bottom, left:right, :]

        return (
            _crop(short_img),
            _crop(long_img),
            _crop(short_raw),
            _crop(long_raw),
        )

    def __getitem__(self, index: int) -> dict:
        pair_idx = index // self.samples_per_pair
        entry = self.entries[pair_idx]

        short_key = entry["short_key"]
        long_key = entry["long_key"]
        expo_ratio = float(entry["exposure_ratio"])
        pair_id = entry["pair_id"]

        short_obs = self._fetch_png(short_key, client_key="short")
        long_gt = self._fetch_png(long_key, client_key="long")

        short_raw = short_obs.astype(np.float32) / MAX_16BIT_VALUE
        long_raw = long_gt.astype(np.float32) / MAX_16BIT_VALUE

        aligned_short: np.ndarray = np.clip(short_raw * expo_ratio, 0.0, 1.0)
        short_srgb = aligned_short
        long_srgb = long_raw

        short_srgb, long_srgb, short_raw, long_raw = self._maybe_random_crop(
            short_srgb, long_srgb, short_raw, long_raw
        )

        tensor_short = img2tensor(short_srgb, bgr2rgb=False, float32=True)
        tensor_long = img2tensor(long_srgb, bgr2rgb=False, float32=True)
        tensor_short_raw = img2tensor(short_raw, bgr2rgb=False, float32=True)
        tensor_long_raw = img2tensor(long_raw, bgr2rgb=False, float32=True)

        sample = {
            "lq": tensor_short,
            "gt": tensor_long,
            "short": tensor_short,
            "long": tensor_long,
            "short_raw": tensor_short_raw,
            "long_raw": tensor_long_raw,
            # short_obs alignment matches the sRGB input tensor
            "short_obs": tensor_short,
            "expo_ratio": torch.full((1, 1, 1), expo_ratio, dtype=torch.float32),
            "pair_id": pair_id,
            "lq_path": short_key,
            "gt_path": long_key,
            "key": str(pair_id),
        }

        if self.return_metadata:
            sample["metadata"] = {
                "pair_id": pair_id,
                "short_key": short_key,
                "long_key": long_key,
                "subset": entry.get("subset"),
                "short_exposure": entry.get("short_exposure"),
                "long_exposure": entry.get("long_exposure"),
                "exposure_ratio": expo_ratio,
            }

        return sample

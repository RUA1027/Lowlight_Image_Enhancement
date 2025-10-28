from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rawpy  # type: ignore
except ImportError as exc:  # pragma: no cover - surfaced during runtime usage
    raise ImportError(
        "rawpy is required to read SID RAW files. Install it via `pip install rawpy`."
    ) from exc

LOGGER = logging.getLogger(__name__)

RAW_EXTENSIONS = (".ARW", ".arw")
MAX_16BIT_VALUE = np.float32(65535.0)
_EXPOSURE_PATTERN = re.compile(r"(?P<Value>\d+(?:\.\d+)?)(?P<Unit>s|ms)$", re.IGNORECASE)


@dataclass(frozen=True)
class SIDPairMetadata:
    """Metadata that describes a matched short/long exposure pair."""

    pair_id: str
    short_path: Path
    long_path: Path
    short_exposure: float
    long_exposure: float

    @property
    def exposure_ratio(self) -> float:
        """Return the ratio required to align short exposure brightness to the long exposure."""
        if self.short_exposure <= 0.0:
            raise ValueError(f"Short exposure for pair {self.pair_id} must be positive.")
        return self.long_exposure / self.short_exposure


def _iter_raw_files(directory: Path) -> Iterable[Path]:
    for extension in RAW_EXTENSIONS:
        yield from directory.glob(f"*{extension}")


def _parse_sid_filename(path: Path) -> Tuple[str, float]:
    """Extract the pair identifier and exposure time (seconds) from a SID filename."""
    stem = path.stem
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Unexpected SID filename format: {path.name}")

    pair_id = "_".join(parts[:2])
    exposure_token = parts[2]

    match = _EXPOSURE_PATTERN.match(exposure_token)
    if not match:
        raise ValueError(f"Unable to parse exposure from filename: {path.name}")

    exposure_value = float(match.group("Value"))
    unit = match.group("Unit").lower()

    if unit == "ms":
        exposure_value /= 1000.0

    if exposure_value <= 0.0:
        raise ValueError(f"Exposure must be positive in filename: {path.name}")

    return pair_id, exposure_value


def find_sid_pairs(
    root_dir: Union[str, os.PathLike[str]],
    camera: str = "Sony",
    allow_incomplete: bool = False,
) -> List[SIDPairMetadata]:
    """Scan the SID directory and build matched long/short exposure pairs.

    Args:
        root_dir: Root directory of the SID dataset (the folder that contains
            the camera subdirectories such as ``Sony`` and ``Fuji``).
        camera: Camera subset to use, defaults to ``"Sony"``.
        allow_incomplete: If ``False`` (default), only perfect long/short matches are
            returned. If ``True``, unmatched entries are ignored instead of raising.

    Returns:
        A list of :class:`SIDPairMetadata` entries sorted by pair id.
    """

    root = Path(root_dir)
    camera_dir = root / camera
    long_dir = camera_dir / "long"
    short_dir = camera_dir / "short"

    if not long_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {long_dir}")
    if not short_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {short_dir}")

    short_records: Dict[str, Tuple[Path, float]] = {}
    for short_path in _iter_raw_files(short_dir):
        pair_id, exposure = _parse_sid_filename(short_path)
        if pair_id in short_records:
            LOGGER.warning(
                "Duplicate short exposure for %s detected. Keeping %s, ignoring %s",
                pair_id,
                short_records[pair_id][0].name,
                short_path.name,
            )
            continue
        short_records[pair_id] = (short_path, exposure)

    long_records: Dict[str, Tuple[Path, float]] = {}
    for long_path in _iter_raw_files(long_dir):
        pair_id, exposure = _parse_sid_filename(long_path)
        if pair_id in long_records:
            LOGGER.warning(
                "Duplicate long exposure for %s detected. Keeping %s, ignoring %s",
                pair_id,
                long_records[pair_id][0].name,
                long_path.name,
            )
            continue
        long_records[pair_id] = (long_path, exposure)

    common_ids = sorted(set(short_records).intersection(long_records))
    missing_in_short = sorted(set(long_records) - set(short_records))
    missing_in_long = sorted(set(short_records) - set(long_records))

    if not allow_incomplete:
        if missing_in_short:
            raise FileNotFoundError(
                f"{len(missing_in_short)} long exposures have no matching short exposure. "
                f"Examples: {missing_in_short[:5]}"
            )
        if missing_in_long:
            raise FileNotFoundError(
                f"{len(missing_in_long)} short exposures have no matching long exposure. "
                f"Examples: {missing_in_long[:5]}"
            )
    else:
        if missing_in_short:
            LOGGER.warning(
                "%d long exposures skipped because no short exposure was found.",
                len(missing_in_short),
            )
        if missing_in_long:
            LOGGER.warning(
                "%d short exposures skipped because no long exposure was found.",
                len(missing_in_long),
            )

    pairs = [
        SIDPairMetadata(
            pair_id=pair_id,
            short_path=short_records[pair_id][0],
            long_path=long_records[pair_id][0],
            short_exposure=short_records[pair_id][1],
            long_exposure=long_records[pair_id][1],
        )
        for pair_id in common_ids
    ]

    if not pairs:
        raise RuntimeError(
            f"No SID pairs discovered under {camera_dir}. "
            "Ensure the dataset is downloaded and unzipped correctly."
        )

    return pairs


class SonySIDDataset(Dataset):
    """PyTorch dataset for the Sony SID subset with RAW preprocessing.

    This dataset performs the complete preprocessing pipeline described in the
    project notes:

    * Reads RAW ``.ARW`` files using ``rawpy`` with 16-bit output and camera white balance.
    * Aligns the short exposure brightness via the exposure ratio parsed from the filename.
    * Clips values to the 16-bit range, normalises to ``[0, 1]`` and returns ``torch.FloatTensor``.
    * Optionally samples random aligned patches to reduce memory usage during training.
    """

    def __init__(
        self,
        root_dir: Union[str, os.PathLike[str]],
        camera: str = "Sony",
        patch_size: Optional[int] = 512,
        random_crop: bool = True,
        samples_per_pair: int = 1,
        cache_in_memory: bool = False,
        rng_seed: Optional[int] = None,
        return_metadata: bool = False,
        allowed_pair_ids: Optional[Sequence[str]] = None,
        allow_incomplete: bool = False,
    ) -> None:
        """Initialise the dataset.

        Args:
            root_dir: Root directory that contains the SID dataset folders.
            camera: Which camera subset to use (default: ``"Sony"``).
            patch_size: Square patch size to crop from each image. If ``None``, the full
                resolution image is returned.
            random_crop: When ``True`` and ``patch_size`` is not ``None``, randomly sample
                crop locations for data augmentation. Otherwise, a centred crop is used.
            samples_per_pair: Logical multiplier that lets the dataset return multiple
                patches per long/short pair within an epoch. Must be >= 1.
            cache_in_memory: Whether to cache the processed 16-bit RGB arrays in memory
                to avoid repeated RAW decoding. Useful for small experiments at the cost
                of increased RAM usage.
            rng_seed: Seed for deterministic cropping.
            return_metadata: If ``True``, ``__getitem__`` returns ``(input, target, meta)``.
            allowed_pair_ids: Optional whitelist of pair identifiers. When provided, only
                these pairs are kept.
            allow_incomplete: Forwarded to :func:`find_sid_pairs`.
        """

        if samples_per_pair < 1:
            raise ValueError("samples_per_pair must be >= 1.")

        self.root_dir = Path(root_dir)
        self.camera = camera
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.samples_per_pair = int(samples_per_pair)
        self.cache_in_memory = cache_in_memory
        self.return_metadata = return_metadata
        self._rng = np.random.default_rng(rng_seed) if rng_seed is not None else np.random.default_rng()

        all_pairs = find_sid_pairs(
            self.root_dir,
            camera=self.camera,
            allow_incomplete=allow_incomplete,
        )

        if allowed_pair_ids is not None:
            allowed = set(allowed_pair_ids)
            filtered_pairs = [pair for pair in all_pairs if pair.pair_id in allowed]
            missing = allowed - {pair.pair_id for pair in filtered_pairs}
            if missing:
                raise ValueError(f"Requested pair ids not found in dataset: {sorted(missing)}")
            self.pairs = filtered_pairs
        else:
            self.pairs = all_pairs

        if not self.pairs:
            raise RuntimeError("No SID pairs available after applying filters.")

        self._cache: Dict[Path, np.ndarray] = {}
        self._image_shape: Optional[Tuple[int, int, int]] = None

    def __len__(self) -> int:
        return len(self.pairs) * self.samples_per_pair

    def __getitem__(self, index: int):
        pair_idx = index // self.samples_per_pair
        pair = self.pairs[pair_idx]

        long_rgb = self._load_rgb_uint16(pair.long_path)
        short_rgb = self._load_rgb_uint16(pair.short_path)

        long_float = long_rgb.astype(np.float32)
        short_float = short_rgb.astype(np.float32)

        aligned_short = np.clip(short_float * pair.exposure_ratio, 0.0, MAX_16BIT_VALUE)

        input_image = (aligned_short / MAX_16BIT_VALUE).astype(np.float32, copy=False)
        target_image = (long_float / MAX_16BIT_VALUE).astype(np.float32, copy=False)

        if self.patch_size is not None:
            input_image, target_image = self._crop_pair(input_image, target_image)

        input_tensor = torch.from_numpy(np.ascontiguousarray(input_image.transpose(2, 0, 1)))
        target_tensor = torch.from_numpy(np.ascontiguousarray(target_image.transpose(2, 0, 1)))

        if self.return_metadata:
            metadata = {
                "pair_id": pair.pair_id,
                "short_path": str(pair.short_path),
                "long_path": str(pair.long_path),
                "short_exposure": pair.short_exposure,
                "long_exposure": pair.long_exposure,
                "exposure_ratio": pair.exposure_ratio,
            }
            return input_tensor, target_tensor, metadata

        return input_tensor, target_tensor

    def _load_rgb_uint16(self, path: Path) -> np.ndarray:
        if self.cache_in_memory and path in self._cache:
            return self._cache[path]

        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=True,
                output_bps=16,
            )

        if rgb.dtype != np.uint16:
            raise RuntimeError(f"Expected uint16 output from rawpy, got {rgb.dtype}")

        if self.cache_in_memory:
            self._cache[path] = rgb

        if self._image_shape is None:
            self._image_shape = rgb.shape  # type: ignore[assignment]

        return rgb

    def _crop_pair(self, input_img: np.ndarray, target_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if input_img.shape != target_img.shape:
            raise ValueError("Input and target images must share the same shape before cropping.")

        h, w, _ = input_img.shape
        patch = self.patch_size
        if patch is None:
            return input_img, target_img

        if patch > h or patch > w:
            raise ValueError(
                f"Requested patch_size={patch} exceeds image dimensions ({h}x{w}). "
                "Reduce the patch size or disable cropping."
            )

        if self.random_crop:
            top = int(self._rng.integers(0, h - patch + 1))
            left = int(self._rng.integers(0, w - patch + 1))
        else:
            top = (h - patch) // 2
            left = (w - patch) // 2

        bottom = top + patch
        right = left + patch

        return (
            input_img[top:bottom, left:right, :],
            target_img[top:bottom, left:right, :],
        )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        patch = self.patch_size if self.patch_size is not None else "full"
        return (
            f"SonySIDDataset(num_pairs={len(self.pairs)}, camera='{self.camera}', "
            f"patch={patch}, samples_per_pair={self.samples_per_pair}, cache={self.cache_in_memory})"
        )

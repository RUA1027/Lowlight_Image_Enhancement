"""Build a manifest describing Sony SID short/long pairs for LMDB loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from datasets.sony_sid_dataset import _parse_sid_filename  # type: ignore


def _collect_pairs(root: Path) -> Dict[str, Tuple[Path, float]]:
    records: Dict[str, Tuple[Path, float]] = {}
    for path in sorted(root.rglob("*.png")):
        pair_id, exposure = _parse_sid_filename(path)
        records[pair_id] = (path, exposure)
    return records


def _relative_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.as_posix()


def build_manifest(
    short_root: Path,
    long_root: Path,
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    split_file: Path | None = None,
) -> List[Dict]:
    short_records = _collect_pairs(short_root)
    long_records = _collect_pairs(long_root)
    common_ids = sorted(set(short_records) & set(long_records))
    if not common_ids:
        raise RuntimeError("No matching short/long pairs were found.")

    if split_file and split_file.is_file():
        with split_file.open("r", encoding="utf-8") as f:
            split_spec = json.load(f)
        splits = {subset: list(split_spec.get(subset, [])) for subset in ("train", "val", "test")}
    else:
        import numpy as np

        rng = np.random.default_rng(seed)
        permuted = rng.permutation(common_ids)
        total = len(permuted)
        num_val = int(total * val_ratio)
        num_test = int(total * test_ratio)
        num_train = total - num_val - num_test
        splits = {
            "train": [str(permuted[i]) for i in range(num_train)],
            "val": [str(permuted[i]) for i in range(num_train, num_train + num_val)],
            "test": [str(permuted[i]) for i in range(num_train + num_val, total)],
        }

    manifest: List[Dict] = []
    for subset, pair_ids in splits.items():
        for pair_id in pair_ids:
            if pair_id not in short_records or pair_id not in long_records:
                continue
            short_path, short_expo = short_records[pair_id]
            long_path, long_expo = long_records[pair_id]
            ratio = long_expo / short_expo
            manifest.append(
                {
                    "pair_id": pair_id,
                    "subset": subset,
                    "short_key": _relative_key(short_path, short_root),
                    "long_key": _relative_key(long_path, long_root),
                    "short_exposure": short_expo,
                    "long_exposure": long_expo,
                    "exposure_ratio": ratio,
                }
            )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SID manifest for LMDB/dataloader usage.")
    parser.add_argument("--short-root", type=Path, required=True, help="Directory with aligned short PNG files.")
    parser.add_argument("--long-root", type=Path, required=True, help="Directory with long PNG files.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON manifest path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting (default: 42).")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio (default: 0.1).")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio (default: 0.1).")
    parser.add_argument("--split-file", type=Path, default=None, help="Optional JSON file with train/val/test pair IDs.")
    args = parser.parse_args()

    manifest = build_manifest(
        args.short_root.resolve(),
        args.long_root.resolve(),
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_file=args.split_file,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {args.output} (total entries: {len(manifest)})")


if __name__ == "__main__":
    main()

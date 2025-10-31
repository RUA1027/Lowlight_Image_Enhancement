"""Convert Sony SID RAW (.ARW) files to 16-bit PNG for LMDB preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import rawpy  # type: ignore
from tqdm import tqdm


def convert_sid_split(raw_dir: Path, png_dir: Path, *, compress_level: int) -> None:
    png_dir.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(raw_dir.glob("*.ARW"))
    for raw_path in tqdm(raw_files, desc=f"{raw_dir.name}->{png_dir.name}"):
        with rawpy.imread(str(raw_path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=True,
                output_bps=16,
            )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out_path = png_dir / (raw_path.stem + ".png")
        cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SID RAW images to 16-bit PNG.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Root containing 'short' and 'long' subfolders with ARW files.")
    parser.add_argument("--output-root", type=Path, required=True, help="Destination root for PNG files (mirrors subfolders).")
    parser.add_argument("--compress-level", type=int, default=1, help="PNG compression level (default 1).")
    args = parser.parse_args()

    short_raw = args.raw_root / "short"
    long_raw = args.raw_root / "long"
    if not short_raw.is_dir() or not long_raw.is_dir():
        raise FileNotFoundError("Expected 'short' and 'long' directories containing ARW files.")

    convert_sid_split(short_raw, args.output_root / "short", compress_level=args.compress_level)
    convert_sid_split(long_raw, args.output_root / "long", compress_level=args.compress_level)


if __name__ == "__main__":
    main()

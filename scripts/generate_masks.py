"""Precompute segmentation pseudo-masks into data/masks.

SegmentationDataset regenerates an Otsu pseudo-mask on every __getitem__, which
means the same OpenCV work is repeated once per image per epoch and leaves the
GPU waiting on the data loader. Running this once writes the masks to disk so
training can just read them.

    python -m scripts.generate_masks --config configs/config_processed.yaml

A note on what these masks are: generate_pseudo_mask applies Otsu thresholding
and keeps the largest connected component, so it segments the *brain*, not the
tumour. It is a stand-in for expert annotations, which this dataset does not
ship. Any Dice/IoU reported against it measures brain-region agreement and
should be described that way in the report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from src.data.preprocessing import generate_pseudo_mask
from src.utils.project import CLASS_NAMES, load_config, mask_path_for, resolve_raw_dir

SPLITS = ("Training", "Testing")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute segmentation pseudo-masks")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate masks that already exist instead of skipping them",
    )
    return parser.parse_args()


def generate_split_masks(
    raw_dir: Path,
    mask_dir: Path,
    split: str,
    img_size: int,
    overwrite: bool,
) -> tuple[int, int]:
    """Write masks for one split. Returns (written, skipped)."""
    split_dir = raw_dir / split
    if not split_dir.exists():
        print(f"Skipping missing split: {split_dir}")
        return 0, 0

    image_paths = [
        path
        for class_name in CLASS_NAMES
        if (split_dir / class_name).exists()
        for path in sorted((split_dir / class_name).iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]

    written = 0
    skipped = 0
    for image_path in tqdm(image_paths, desc=f"{split} masks"):
        out_path = mask_path_for(image_path, raw_dir, mask_dir)
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        mask = generate_pseudo_mask(str(image_path), img_size)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), mask):
            raise RuntimeError(f"Failed to write mask: {out_path}")
        written += 1

    return written, skipped


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]

    raw_dir = resolve_raw_dir(config)
    mask_dir = Path(data_cfg["mask_dir"])
    img_size = int(data_cfg["img_size"])

    print(f"Source images: {raw_dir}")
    print(f"Mask output:   {mask_dir}")

    total_written = 0
    total_skipped = 0
    for split in SPLITS:
        written, skipped = generate_split_masks(raw_dir, mask_dir, split, img_size, args.overwrite)
        total_written += written
        total_skipped += skipped

    print(f"Wrote {total_written} masks, skipped {total_skipped} existing.")
    if total_skipped and not args.overwrite:
        print("Pass --overwrite to regenerate the skipped masks.")


if __name__ == "__main__":
    main()

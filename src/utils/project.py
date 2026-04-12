from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_raw_dir(config: dict[str, Any]) -> Path:
    raw_dir = Path(config["data"]["raw_dir"])
    if raw_dir.exists():
        return raw_dir

    fallback = Path("Dataset")
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not find dataset directory. Checked '{raw_dir}' and fallback 'Dataset'."
    )


def list_split_image_paths(raw_dir: Path, split: str) -> tuple[list[str], list[int]]:
    split_dir = raw_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    image_paths: list[str] = []
    labels: list[int] = []

    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        files = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        image_paths.extend([str(p) for p in files])
        labels.extend([CLASS_TO_IDX[class_name]] * len(files))

    return image_paths, labels

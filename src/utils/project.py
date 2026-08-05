from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# keras.applications.EfficientNet* starts with Rescaling(1/255) + Normalization,
# so it expects raw pixels in [0, 255]. Our preprocessing emits [0, 1], and every
# TensorFlow entry point (training, evaluation, inference) must scale back up by
# this factor. The PyTorch U-Net encoder has no built-in preprocessing and keeps
# using explicit ImageNet mean/std normalization instead.
EFFICIENTNET_INPUT_SCALE = 255.0


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_raw_dir(config: dict[str, Any]) -> Path:
    raw_dir = Path(config["data"]["raw_dir"])
    if raw_dir.exists():
        return raw_dir

    # Falling back would silently pair unprocessed images with the decode-only
    # loader, skipping the crop and CLAHE steps the model was configured for.
    if bool(config["data"].get("preprocessed_input", False)):
        raise FileNotFoundError(
            f"data.raw_dir '{raw_dir}' does not exist and preprocessed_input is true. "
            "Generate it first: python -m src.data.preprocessing "
            f"--raw-dir Dataset --out-dir {raw_dir}"
        )

    fallback = Path("Dataset")
    if fallback.exists():
        print(f"data.raw_dir '{raw_dir}' not found; falling back to '{fallback}'")
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


def mask_path_for(image_path: str | Path, raw_dir: str | Path, mask_dir: str | Path) -> Path:
    """Mask counterpart of an image, mirroring the split/class folder layout."""
    relative = Path(image_path).relative_to(Path(raw_dir))
    return Path(mask_dir) / relative.with_suffix(".png")


def mask_paths_for(
    image_paths: list[str], raw_dir: str | Path, mask_dir: str | Path
) -> list[str]:
    """Map a list of image paths to their precomputed mask paths."""
    return [str(mask_path_for(p, raw_dir, mask_dir)) for p in image_paths]

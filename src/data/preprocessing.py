from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.utils.project import CLASS_NAMES


def crop_brain_region(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Crop image to the largest non-background contour.

    Steps:
    1. Convert to grayscale.
    2. Threshold to isolate foreground.
    3. Morphological close/open.
    4. Keep largest contour and crop with padding.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    pad = 10
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
    return image[y1:y2, x1:x2]


def preprocess_image(image_path: str, target_size: int = 224) -> np.ndarray:
    """Return RGB float32 image in [0, 1] with shape (target_size, target_size, 3)."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_brain_region(img)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img.astype(np.float32) / 255.0


def generate_pseudo_mask(image_path: str, target_size: int = 224) -> np.ndarray:
    """Generate a binary pseudo-mask via Otsu thresholding on bright tissue regions."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img = crop_brain_region(img)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 2:
        largest_cc = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = ((labels == largest_cc) * 255).astype(np.uint8)

    return mask


def preprocess_dataset(raw_dir: Path, out_dir: Path, target_size: int = 224) -> None:
    """Preprocess all images under Training/ and Testing/ preserving class folders."""
    splits = ["Training", "Testing"]
    for split in splits:
        for class_name in CLASS_NAMES:
            in_dir = raw_dir / split / class_name
            if not in_dir.exists():
                continue

            dst_dir = out_dir / split / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)

            for image_path in sorted(in_dir.iterdir()):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                out_img = preprocess_image(str(image_path), target_size)
                out_bgr = cv2.cvtColor((out_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(dst_dir / image_path.name), out_bgr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess brain MRI dataset")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Raw dataset root")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--img-size", type=int, default=224, help="Target image size")
    args = parser.parse_args()

    preprocess_dataset(args.raw_dir, args.out_dir, args.img_size)
    print(f"Preprocessing complete: {args.out_dir}")


if __name__ == "__main__":
    main()

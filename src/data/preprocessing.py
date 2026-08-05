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


def load_preprocessed_image(image_path: str, target_size: int = 224) -> np.ndarray:
    """Read an image that has already been through preprocess_image.

    Decodes and resizes only. Re-running crop_brain_region and CLAHE on a file
    written by preprocess_dataset would enhance contrast twice and desync
    training from the inference pipeline.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (target_size, target_size):
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return img.astype(np.float32) / 255.0


def load_image(image_path: str, target_size: int = 224, preprocessed: bool = False) -> np.ndarray:
    """Load an MRI as RGB float32 in [0, 1], preprocessing it only if needed."""
    if preprocessed:
        return load_preprocessed_image(image_path, target_size)
    return preprocess_image(image_path, target_size)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes by flood-filling the background from the border."""
    h, w = mask.shape
    flood = mask.copy()
    scratch = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, scratch, (0, 0), 255)
    return mask | cv2.bitwise_not(flood)


def segment_brain(gray: np.ndarray) -> np.ndarray:
    """Binary mask of the head/brain region: Otsu, largest component, holes filled."""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
    if num_labels <= 1:
        return np.zeros_like(gray, dtype=np.uint8)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    brain = ((labels == largest) * 255).astype(np.uint8)
    return _fill_holes(brain)


def generate_pseudo_mask(
    image_path: str,
    target_size: int = 224,
    hyper_percentile: float = 96.0,
    sigma_above_mean: float = 1.5,
    min_area_frac: float = 0.004,
    max_area_frac: float = 0.30,
    rim_erosion: int = 17,
    min_circularity: float = 0.25,
    min_zscore: float = 2.0,
) -> np.ndarray:
    """Approximate a tumour mask from intensity alone, for use as a weak label.

    On T1-contrast-enhanced MRI a tumour is typically hyperintense relative to
    surrounding brain tissue, so the mask is the brightest compact blob inside the
    brain:

    1. Segment the head, then erode inward. The skull and scalp are bright too and
       would otherwise dominate the "brightest region" search.
    2. Threshold at a high percentile of the *intra-brain* intensity distribution,
       and require the value to also sit well above the tissue mean, so images
       without a bright lesion do not produce a mask by construction.
    3. Keep the highest-scoring connected component, preferring blobs that are
       compact (solid) and plausibly sized relative to the brain.

    Returns an all-zero mask when nothing qualifies, which is the desired answer
    for a scan with no tumour.

    This is a weak label, not ground truth. It cannot see non-enhancing tumour and
    it will occasionally lock onto normal bright structures. For genuine
    segmentation quality, train against a dataset that ships expert annotations.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img = crop_brain_region(img)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    empty = np.zeros((target_size, target_size), dtype=np.uint8)

    brain = segment_brain(img)
    if not brain.any():
        return empty

    # Drop the skull/scalp rim so the search sees brain tissue only.
    rim_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim_erosion, rim_erosion))
    interior = cv2.erode(brain, rim_kernel)
    interior_area = int((interior > 0).sum())
    if interior_area < 200:
        return empty

    values = img[interior > 0]
    threshold = max(
        float(np.percentile(values, hyper_percentile)),
        float(values.mean() + sigma_above_mean * values.std()),
    )

    candidates = ((img >= threshold) & (interior > 0)).astype(np.uint8) * 255
    candidates = cv2.morphologyEx(
        candidates, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    candidates = cv2.morphologyEx(
        candidates, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )

    # Anything hugging the eroded edge is skull or scalp that survived the erosion,
    # not a lesion. This is the single most common false positive.
    edge_band = cv2.subtract(
        interior, cv2.erode(interior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    )

    intensity_mean = float(values.mean())
    intensity_std = float(values.std()) + 1e-6

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidates)
    best_label = -1
    best_score = 0.0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_frac * interior_area or area > max_area_frac * interior_area:
            continue

        component = (labels == label).astype(np.uint8)
        if cv2.countNonZero(cv2.bitwise_and(component * 255, edge_band)) > 0.15 * area:
            continue

        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        # Rim slivers are long and thin; a lesion is roughly blob-shaped.
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0.0
        if circularity < min_circularity:
            continue

        # Require the blob to be a genuine intensity outlier, not just the
        # brightest thing in an image that happens to have no lesion.
        blob_mean = float(img[component > 0].mean())
        if (blob_mean - intensity_mean) / intensity_std < min_zscore:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 0.0

        # Favour large, solid blobs; scattered speckle scores poorly.
        score = area * solidity
        if score > best_score:
            best_score = score
            best_label = label

    if best_label < 0:
        return empty

    mask = ((labels == best_label) * 255).astype(np.uint8)
    return _fill_holes(mask)


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

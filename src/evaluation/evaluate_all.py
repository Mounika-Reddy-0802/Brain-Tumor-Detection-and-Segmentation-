from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import torch
from tqdm import tqdm

from src.data.preprocessing import generate_pseudo_mask, load_image
from src.evaluation.metrics import (
    classification_metrics,
    detection_metrics,
    dice_coefficient,
    iou_score,
)
from src.models.torch_unet import get_unet
from src.utils.gpu_setup import configure_gpu
from src.utils.project import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    EFFICIENTNET_INPUT_SCALE,
    list_split_image_paths,
    load_config,
    mask_paths_for,
    resolve_raw_dir,
)

# Used by the PyTorch U-Net only; the TF models normalize internally.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detection, classification, and segmentation")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    return parser.parse_args()


def load_image_batch(image_paths: list[str], img_size: int, preprocessed: bool) -> np.ndarray:
    """Stack images as (N, H, W, 3) float32 in [0, 1]."""
    return np.stack([load_image(p, img_size, preprocessed) for p in image_paths]).astype(np.float32)


def predict_in_batches(
    model,
    image_paths: list[str],
    img_size: int,
    preprocessed: bool,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    """Run a Keras model over all images, batched.

    Calling model.predict() once per image carries enough per-call overhead to
    dominate the runtime on CPU, so batch the forward passes.
    """
    outputs = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc=desc):
        chunk = image_paths[start : start + batch_size]
        # EfficientNet rescales and normalizes internally, so it wants [0, 255].
        batch = load_image_batch(chunk, img_size, preprocessed) * EFFICIENTNET_INPUT_SCALE
        outputs.append(model.predict(batch, verbose=0))
    return np.concatenate(outputs, axis=0)


def evaluate_detection(
    detector,
    image_paths: list[str],
    labels: np.ndarray,
    img_size: int,
    preprocessed: bool,
    batch_size: int,
) -> dict:
    y_true = np.where(labels == CLASS_TO_IDX["notumor"], 0, 1)
    probs = predict_in_batches(
        detector, image_paths, img_size, preprocessed, batch_size, "Detection evaluation"
    )[:, 0]
    return detection_metrics(y_true, probs.astype(np.float32))


def evaluate_classification(
    classifier,
    image_paths: list[str],
    labels: np.ndarray,
    img_size: int,
    preprocessed: bool,
    batch_size: int,
) -> dict:
    cls_probs = predict_in_batches(
        classifier, image_paths, img_size, preprocessed, batch_size, "Classification evaluation"
    )
    preds = np.argmax(cls_probs, axis=1).astype(np.int64)
    return classification_metrics(labels, preds, CLASS_NAMES)


def load_true_masks(
    image_paths: list[str],
    mask_paths: list[str] | None,
    img_size: int,
) -> np.ndarray:
    """Stack reference masks as (N, H, W) float32 in {0, 1}."""
    masks = []
    for idx, image_path in enumerate(image_paths):
        if mask_paths is None:
            mask = generate_pseudo_mask(image_path, target_size=img_size)
        else:
            mask = cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to read mask: {mask_paths[idx]}")
            if mask.shape != (img_size, img_size):
                mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        masks.append((mask > 127).astype(np.float32))
    return np.stack(masks)


def evaluate_segmentation(
    model,
    image_paths: list[str],
    mask_paths: list[str] | None,
    img_size: int,
    device: torch.device,
    preprocessed: bool,
    batch_size: int,
) -> dict:
    """Dice and IoU averaged per image.

    The reference masks are Otsu pseudo-masks, not expert annotations, and the
    model was trained against masks from the same generator. These numbers say
    how well the U-Net reproduces that thresholding rule, not how well it finds
    tumours. Report them with that caveat.
    """
    model.eval()
    dices = []
    ious = []

    with torch.no_grad():
        for start in tqdm(range(0, len(image_paths), batch_size), desc="Segmentation evaluation"):
            chunk = image_paths[start : start + batch_size]
            chunk_masks = None if mask_paths is None else mask_paths[start : start + batch_size]

            imgs = load_image_batch(chunk, img_size, preprocessed)
            # smp's encoder has no built-in preprocessing, so normalize explicitly.
            imgs = (imgs - IMAGENET_MEAN) / IMAGENET_STD
            img_tensor = torch.from_numpy(imgs.transpose(0, 3, 1, 2)).float().to(device)

            true_masks = load_true_masks(chunk, chunk_masks, img_size)
            true_tensor = torch.from_numpy(true_masks).unsqueeze(1).to(device)

            pred_probs = torch.sigmoid(model(img_tensor))

            # Score one image at a time: the metric helpers flatten their input,
            # so scoring a whole batch would pool pixels across images.
            for i in range(len(chunk)):
                dices.append(dice_coefficient(pred_probs[i : i + 1], true_tensor[i : i + 1]))
                ious.append(iou_score(pred_probs[i : i + 1], true_tensor[i : i + 1]))

    return {
        "dice": float(np.mean(dices)) if dices else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
        "reference_masks": "precomputed pseudo-masks" if mask_paths else "on-the-fly pseudo-masks",
    }


def save_confusion_matrix(cm: list[list[int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]

    configure_gpu()

    detector_path = Path("models/detection_model.keras")
    classifier_path = Path("models/classification_model.keras")
    unet_path = Path("models/best_unet.pth") if Path("models/best_unet.pth").exists() else Path("models/unet_last.pth")

    missing = [p for p in [detector_path, classifier_path, unet_path] if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing trained model files: {missing_str}")

    raw_dir = resolve_raw_dir(config)
    image_paths, labels = list_split_image_paths(raw_dir, "Testing")
    labels_np = np.array(labels, dtype=np.int64)

    detector = tf.keras.models.load_model(detector_path)
    classifier = tf.keras.models.load_model(classifier_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = get_unet(encoder_name=config["segmentation"]["encoder"], pretrained=False).to(device)
    unet.load_state_dict(torch.load(unet_path, map_location=device))

    img_size = int(data_cfg["img_size"])
    preprocessed = bool(data_cfg.get("preprocessed_input", False))
    batch_size = int(config["tensorflow"].get("batch_size", 32))

    mask_paths = None
    if bool(data_cfg.get("precomputed_masks", False)):
        mask_paths = mask_paths_for(image_paths, raw_dir, Path(data_cfg["mask_dir"]))
        missing_masks = [p for p in mask_paths if not Path(p).exists()]
        if missing_masks:
            raise FileNotFoundError(
                f"precomputed_masks is enabled but {len(missing_masks)} mask(s) are missing, "
                f"starting with {missing_masks[0]}. "
                f"Run: python -m scripts.generate_masks --config {args.config}"
            )

    results = {
        "detection": evaluate_detection(
            detector, image_paths, labels_np, img_size, preprocessed, batch_size
        ),
        "classification": evaluate_classification(
            classifier, image_paths, labels_np, img_size, preprocessed, batch_size
        ),
        "segmentation": evaluate_segmentation(
            unet, image_paths, mask_paths, img_size, device, preprocessed, batch_size
        ),
    }

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    cm = results["classification"]["confusion_matrix"]
    save_confusion_matrix(cm, output_dir / "classification_confusion_matrix.png")

    print(json.dumps(results, indent=2))
    print("Saved evaluation report to outputs/evaluation_report.json")


if __name__ == "__main__":
    main()

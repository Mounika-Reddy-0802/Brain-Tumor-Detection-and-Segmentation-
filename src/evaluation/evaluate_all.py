from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import torch
from tqdm import tqdm

from src.data.preprocessing import generate_pseudo_mask, preprocess_image
from src.evaluation.metrics import (
    classification_metrics,
    detection_metrics,
    dice_coefficient,
    iou_score,
)
from src.models.torch_unet import get_unet
from src.utils.gpu_setup import configure_gpu
from src.utils.project import CLASS_NAMES, CLASS_TO_IDX, list_split_image_paths, load_config, resolve_raw_dir

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detection, classification, and segmentation")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    return parser.parse_args()


def to_tf_input(image_path: str, img_size: int) -> np.ndarray:
    img = preprocess_image(image_path, target_size=img_size)
    norm = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(norm, axis=0).astype(np.float32)


def evaluate_detection(detector, image_paths: list[str], labels: np.ndarray, img_size: int) -> dict:
    y_true = np.where(labels == CLASS_TO_IDX["notumor"], 0, 1)
    probs = []
    for path in tqdm(image_paths, desc="Detection evaluation"):
        prob = float(detector.predict(to_tf_input(path, img_size), verbose=0)[0, 0])
        probs.append(prob)
    return detection_metrics(y_true, np.array(probs, dtype=np.float32))


def evaluate_classification(classifier, image_paths: list[str], labels: np.ndarray, img_size: int) -> dict:
    preds = []
    for path in tqdm(image_paths, desc="Classification evaluation"):
        cls_probs = classifier.predict(to_tf_input(path, img_size), verbose=0)[0]
        preds.append(int(np.argmax(cls_probs)))
    return classification_metrics(labels, np.array(preds, dtype=np.int64), CLASS_NAMES)


def evaluate_segmentation(model, image_paths: list[str], img_size: int, device: torch.device) -> dict:
    model.eval()
    dices = []
    ious = []

    with torch.no_grad():
        for path in tqdm(image_paths, desc="Segmentation evaluation"):
            img = preprocess_image(path, target_size=img_size)
            mask_np = (generate_pseudo_mask(path, target_size=img_size) > 127).astype(np.float32)

            img_norm = (img - IMAGENET_MEAN) / IMAGENET_STD
            img_tensor = torch.FloatTensor(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
            true_mask = torch.FloatTensor(mask_np).unsqueeze(0).unsqueeze(0).to(device)

            pred_logits = model(img_tensor)
            pred_probs = torch.sigmoid(pred_logits)

            dices.append(dice_coefficient(pred_probs, true_mask))
            ious.append(iou_score(pred_probs, true_mask))

    return {
        "dice": float(np.mean(dices)) if dices else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
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
    results = {
        "detection": evaluate_detection(detector, image_paths, labels_np, img_size),
        "classification": evaluate_classification(classifier, image_paths, labels_np, img_size),
        "segmentation": evaluate_segmentation(unet, image_paths, img_size, device),
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

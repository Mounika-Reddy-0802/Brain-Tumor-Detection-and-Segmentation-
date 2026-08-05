from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from src.data.tf_dataset import build_tf_dataset
from src.models.tf_detection_model import build_detection_model
from src.utils.gpu_setup import configure_gpu
from src.utils.project import CLASS_TO_IDX, list_split_image_paths, load_config, resolve_raw_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TensorFlow binary detection model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_cfg = config["data"]
    model_cfg = config["model"]
    tf_cfg = config["tensorflow"]

    configure_gpu()
    tf_gpus = tf.config.list_physical_devices("GPU")
    require_gpu = bool(tf_cfg.get("require_gpu", True))
    if require_gpu and not tf_gpus:
        raise RuntimeError(
            "No TensorFlow GPU detected, but tensorflow.require_gpu=true. "
            "Run this script in a GPU-enabled TensorFlow environment (for Windows, use WSL2)."
        )

    if tf_cfg.get("mixed_precision", True) and tf_gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    raw_dir = resolve_raw_dir(config)
    image_paths, labels = list_split_image_paths(raw_dir, "Training")

    image_paths = np.array(image_paths)
    labels = np.array(labels)
    y_detection = np.where(labels == CLASS_TO_IDX["notumor"], 0, 1)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(data_cfg.get("val_split", 0.15)),
        random_state=42,
    )
    train_idx, val_idx = next(splitter.split(image_paths, y_detection))

    x_train, x_val = image_paths[train_idx], image_paths[val_idx]
    y_train, y_val = y_detection[train_idx], y_detection[val_idx]

    preprocessed = bool(data_cfg.get("preprocessed_input", False))
    train_ds = build_tf_dataset(
        x_train,
        y_train,
        batch_size=int(tf_cfg["batch_size"]),
        augment=True,
        shuffle=True,
        img_size=int(data_cfg["img_size"]),
        preprocessed=preprocessed,
    )
    val_ds = build_tf_dataset(
        x_val,
        y_val,
        batch_size=int(tf_cfg["batch_size"]),
        augment=False,
        shuffle=False,
        img_size=int(data_cfg["img_size"]),
        preprocessed=preprocessed,
    )

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: float(class_weights[0]), 1: float(class_weights[1])}

    model = build_detection_model(
        img_size=int(data_cfg["img_size"]),
        dropout=float(model_cfg["dropout"]),
    )

    warmup_lr = float(tf_cfg.get("warmup_lr", 1e-3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=warmup_lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.fit(
        train_ds,
        epochs=int(tf_cfg.get("freeze_backbone_epochs", 5)),
        validation_data=val_ds,
        class_weight=class_weight_dict,
        verbose=1,
    )

    model.get_layer("efficientnetb3").trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(tf_cfg["lr"]),
            weight_decay=float(tf_cfg["weight_decay"]),
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("logs/detection").mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall",
            patience=int(tf_cfg["early_stopping_patience"]),
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "checkpoints/best_detection.keras",
            save_best_only=True,
            monitor="val_recall",
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(tf_cfg["reduce_lr_factor"]),
            patience=int(tf_cfg["reduce_lr_patience"]),
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir="logs/detection"),
    ]

    model.fit(
        train_ds,
        epochs=int(tf_cfg["detection_epochs"]),
        validation_data=val_ds,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    model.save("models/detection_model.keras")
    print("Saved: models/detection_model.keras")


if __name__ == "__main__":
    main()

"""Fast pre-flight check for the whole pipeline.

Run this before starting a GPU session. It builds every model, pushes a tiny
batch through each training path and verifies the input conventions, in well
under a minute — so configuration and environment errors surface immediately
instead of twenty minutes into a training run.

    python -m scripts.smoke_test --config configs/config_processed.yaml

Every check here exists because something actually broke:

* the smp encoder registry uses "efficientnet-b3", not "efficientnet_b3"
* Keras 3.11 cannot load ImageNet weights into the V1 EfficientNets by index
* EfficientNet expects [0, 255] input because it rescales internally
* albumentations 2.x silently ignores removed arguments such as var_limit
* multi-class AUC/Precision/Recall metrics reject integer labels
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str):
    """Decorator that records a check's outcome instead of aborting the run."""

    def wrapper(fn):
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001 - report every failure, not just the first
            FAILED.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")
        else:
            PASSED.append(name)
            print(f"  ok    {name}" + (f" - {detail}" if detail else ""))
        return fn

    return wrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-flight checks for the training pipeline")
    parser.add_argument("--config", type=str, default="configs/config_processed.yaml")
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Skip ImageNet weight loading (avoids a 44 MB download when offline)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import tensorflow as tf
    import torch

    from src.utils.project import (
        EFFICIENTNET_INPUT_SCALE,
        list_split_image_paths,
        load_config,
        mask_paths_for,
        resolve_raw_dir,
    )

    config = load_config(args.config)
    data_cfg = config["data"]
    img_size = int(data_cfg["img_size"])
    preprocessed = bool(data_cfg.get("preprocessed_input", False))

    print(f"\nconfig: {args.config}")
    print(f"TF {tf.__version__} | torch {torch.__version__} | "
          f"GPU: TF={len(tf.config.list_physical_devices('GPU'))} torch={torch.cuda.is_available()}")

    print("\n[data]")

    @check("dataset resolves and splits are readable")
    def _data():
        raw_dir = resolve_raw_dir(config)
        paths, labels = list_split_image_paths(raw_dir, "Training")
        assert len(paths) == len(labels) and paths, "empty Training split"
        return f"{len(paths)} training images under {raw_dir}"

    raw_dir = resolve_raw_dir(config)
    train_paths, train_labels = list_split_image_paths(raw_dir, "Training")
    sample_paths = np.array(train_paths[:8])
    sample_labels = np.array(train_labels[:8])

    @check("tf.data pipeline emits [0, 255] for EfficientNet")
    def _pipeline():
        from src.data.tf_dataset import build_tf_dataset

        ds = build_tf_dataset(sample_paths, sample_labels, batch_size=4, augment=True,
                              shuffle=True, img_size=img_size, preprocessed=preprocessed)
        images, _ = next(iter(ds))
        images = images.numpy()
        assert images.shape == (4, img_size, img_size, 3), images.shape
        assert images.max() > 1.5, f"max {images.max():.3f} - looks pre-normalised, not [0, 255]"
        assert images.max() <= 255.001, images.max()
        return f"batch {images.shape}, range [{images.min():.1f}, {images.max():.1f}]"

    @check("albumentations transforms accept every configured argument")
    def _augment():
        from src.data.augmentation import get_train_transforms

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transform = get_train_transforms()
            out = transform(image=np.zeros((img_size, img_size, 3), np.uint8),
                            mask=np.zeros((img_size, img_size), np.float32))
        ignored = [str(w.message) for w in caught if "not valid for transform" in str(w.message)]
        assert not ignored, f"arguments silently ignored: {ignored}"
        assert out["image"].shape == (img_size, img_size, 3)
        return "no ignored arguments"

    print("\n[tensorflow]")

    pretrained = not args.skip_weights

    @check("detection model builds and trains one step")
    def _detection():
        from src.data.tf_dataset import build_tf_dataset
        from src.models.tf_detection_model import build_detection_model
        from src.utils.project import CLASS_TO_IDX

        binary = np.where(sample_labels == CLASS_TO_IDX["notumor"], 0, 1)
        ds = build_tf_dataset(sample_paths, binary, batch_size=4, augment=False,
                              shuffle=False, img_size=img_size, preprocessed=preprocessed)
        model = build_detection_model(img_size=img_size, dropout=0.4)
        model.get_layer("efficientnetb3").trainable = True   # what the trainer does
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss="binary_crossentropy",
                      metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                               tf.keras.metrics.Recall(name="recall")])
        model.fit(ds, epochs=1, verbose=0)
        return f"output {model.output_shape}"

    @check("classification model builds and trains one step")
    def _classification():
        from src.data.tf_dataset import build_tf_dataset
        from src.models.tf_classification_model import build_classification_model

        ds = build_tf_dataset(sample_paths, sample_labels, batch_size=4, augment=False,
                              shuffle=False, img_size=img_size, preprocessed=preprocessed)
        model = build_classification_model(num_classes=4, img_size=img_size, dropout=0.4)
        model.get_layer("efficientnetb3").trainable = True
        # Integer labels: AUC/Precision/Recall would raise a shape error here.
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss="sparse_categorical_crossentropy",
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                               tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2")])
        model.fit(ds, epochs=1, verbose=0)
        return f"output {model.output_shape}"

    if pretrained:
        @check("EfficientNet receives a properly scaled signal")
        def _scaling():
            import keras

            from src.data.preprocessing import load_image
            from src.models.backbone import build_efficientnet_b3

            backbone = build_efficientnet_b3(img_size=img_size, pretrained=True)
            stem_input = keras.Model(backbone.inputs, backbone.get_layer("stem_conv_pad").input)
            image = load_image(str(sample_paths[0]), img_size, preprocessed)
            activations = stem_input.predict(image[None] * EFFICIENTNET_INPUT_SCALE, verbose=0)
            assert activations.std() > 0.02, (
                f"stem sees std={activations.std():.4f} - input range is probably wrong"
            )
            return f"stem std {activations.std():.3f}"

    print("\n[pytorch]")

    @check("segmentation encoder name is a valid smp registry key")
    def _encoder():
        import segmentation_models_pytorch as smp

        encoder = config["segmentation"]["encoder"]
        valid = set(smp.encoders.get_encoder_names())
        assert encoder in valid, (
            f"'{encoder}' is not an smp encoder. Did you use an underscore? "
            f"Closest: {[v for v in valid if v.replace('-', '_') == encoder.replace('-', '_')]}"
        )
        return f"encoder '{encoder}'"

    @check("U-Net builds and the loss decreases over a few steps")
    def _unet():
        import torch as _torch

        from src.data.torch_dataset import SegmentationDataset
        from src.models.torch_unet import get_unet
        from src.training.train_torch_segmentation import CombinedSegmentationLoss

        model = get_unet(encoder_name=config["segmentation"]["encoder"], pretrained=False)
        dataset = SegmentationDataset(
            [str(p) for p in sample_paths[:4]],
            mask_paths=None,
            use_pseudo_masks=True,
            transform=None,
            img_size=img_size,
            preprocessed=preprocessed,
        )
        images = _torch.stack([dataset[i][0] for i in range(4)])
        masks = _torch.stack([dataset[i][1] for i in range(4)])
        assert images.shape == (4, 3, img_size, img_size), images.shape

        criterion = CombinedSegmentationLoss(
            dice_weight=float(config["segmentation"]["dice_weight"]),
            bce_weight=float(config["segmentation"]["bce_weight"]),
        )
        optimizer = _torch.optim.Adam(model.parameters(), lr=1e-4)
        model.train()
        before = float(criterion(model(images), masks).item())
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
        after = float(criterion(model(images), masks).item())
        assert np.isfinite(after) and after < before, f"loss {before:.4f} -> {after:.4f}"
        return f"loss {before:.4f} -> {after:.4f}"

    if bool(data_cfg.get("precomputed_masks", False)):
        @check("precomputed masks exist for the training split")
        def _masks():
            paths = mask_paths_for(train_paths, raw_dir, Path(data_cfg["mask_dir"]))
            missing = [p for p in paths if not Path(p).exists()]
            assert not missing, (
                f"{len(missing)} missing, first: {missing[0]}. "
                f"Run: python -m scripts.generate_masks --config {args.config}"
            )
            return f"{len(paths)} masks present"

    print(f"\n{'=' * 58}")
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED, {len(PASSED)} passed:\n")
        for failure in FAILED:
            print(f"  - {failure}")
        print("\nFix these before starting a GPU run.")
        return 1

    print(f"All {len(PASSED)} checks passed. Safe to start training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

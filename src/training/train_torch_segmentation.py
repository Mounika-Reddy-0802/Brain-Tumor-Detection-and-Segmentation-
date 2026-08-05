from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data.augmentation import get_eval_transforms, get_train_transforms
from src.data.torch_dataset import SegmentationDataset
from src.models.torch_unet import get_unet
from src.utils.gpu_setup import configure_gpu
from src.utils.project import (
    CLASS_NAMES,
    list_split_image_paths,
    load_config,
    mask_paths_for,
    resolve_raw_dir,
)


class CombinedSegmentationLoss(nn.Module):
    """60% Dice loss + 40% BCEWithLogits loss."""

    def __init__(self, dice_weight: float = 0.6, bce_weight: float = 0.4) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice_loss(pred, target) + self.bce_weight * self.bce(pred, target)


class EarlyStopping:
    def __init__(self, patience: int = 8, delta: float = 1e-4) -> None:
        self.patience = patience
        self.delta = delta
        self.best = None
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module, path: str) -> None:
        if self.best is None or val_loss < self.best - self.delta:
            self.best = val_loss
            self.counter = 0
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float().view(-1)
    true = targets.float().view(-1)
    intersection = (preds * true).sum()
    return float((2 * intersection + smooth) / (preds.sum() + true.sum() + smooth))


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            with autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
            total_loss += float(loss.item())
            total_dice += dice_coefficient(logits, masks)
            total_batches += 1

    if total_batches == 0:
        return 0.0, 0.0

    return total_loss / total_batches, total_dice / total_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PyTorch U-Net segmentation model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_cfg = config["data"]
    model_cfg = config["model"]
    pt_cfg = config["pytorch"]
    seg_cfg = config["segmentation"]

    configure_gpu()

    raw_dir = resolve_raw_dir(config)
    image_paths, labels = list_split_image_paths(raw_dir, "Training")

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(data_cfg.get("val_split", 0.15)),
        random_state=42,
    )
    train_idx, val_idx = next(splitter.split(image_paths, labels))

    x_train, x_val = image_paths[train_idx].tolist(), image_paths[val_idx].tolist()
    y_train = labels[train_idx]

    class_counts = np.bincount(y_train, minlength=len(CLASS_NAMES))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [float(class_weights[label]) for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    img_size = int(data_cfg["img_size"])
    use_pseudo_masks = bool(data_cfg.get("use_pseudo_masks", True))
    preprocessed = bool(data_cfg.get("preprocessed_input", False))

    # Reading precomputed masks avoids re-running Otsu on every image every
    # epoch, which is what otherwise keeps the GPU waiting on the data loader.
    train_masks: list[str] | None = None
    val_masks: list[str] | None = None
    if bool(data_cfg.get("precomputed_masks", False)):
        mask_dir = Path(data_cfg["mask_dir"])
        train_masks = mask_paths_for(x_train, raw_dir, mask_dir)
        val_masks = mask_paths_for(x_val, raw_dir, mask_dir)

        missing = [p for p in train_masks + val_masks if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"precomputed_masks is enabled but {len(missing)} mask(s) are missing, "
                f"starting with {missing[0]}. "
                f"Run: python -m scripts.generate_masks --config {args.config}"
            )
        print(f"Using {len(train_masks) + len(val_masks)} precomputed masks from {mask_dir}")
    else:
        print("Generating pseudo-masks on the fly (slower; see scripts/generate_masks.py)")

    train_dataset = SegmentationDataset(
        image_paths=x_train,
        mask_paths=train_masks,
        use_pseudo_masks=use_pseudo_masks,
        transform=get_train_transforms(),
        img_size=img_size,
        preprocessed=preprocessed,
    )
    val_dataset = SegmentationDataset(
        image_paths=x_val,
        mask_paths=val_masks,
        use_pseudo_masks=use_pseudo_masks,
        transform=get_eval_transforms(),
        img_size=img_size,
        preprocessed=preprocessed,
    )

    batch_size = int(pt_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(pt_cfg.get("mixed_precision", True) and device.type == "cuda")

    model = get_unet(
        encoder_name=str(seg_cfg.get("encoder", "efficientnet_b3")),
        pretrained=bool(model_cfg.get("pretrained", True)),
    ).to(device)

    criterion = CombinedSegmentationLoss(
        dice_weight=float(seg_cfg.get("dice_weight", 0.6)),
        bce_weight=float(seg_cfg.get("bce_weight", 0.4)),
    )
    optimizer = Adam(
        model.parameters(),
        lr=float(pt_cfg["lr"]),
        weight_decay=float(pt_cfg["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(pt_cfg.get("T_max", pt_cfg["segmentation_epochs"])),
        eta_min=1e-6,
    )
    scaler = GradScaler("cuda", enabled=amp_enabled)

    early_stopping = EarlyStopping(patience=int(pt_cfg["early_stopping_patience"]))

    max_epochs = int(pt_cfg["segmentation_epochs"])
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            num_batches += 1

        scheduler.step()
        train_loss = running_loss / max(num_batches, 1)
        val_loss, val_dice = validate(model, val_loader, criterion, device, amp_enabled)

        print(
            f"Epoch {epoch + 1:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
        )

        early_stopping(val_loss, model, "models/best_unet.pth")
        if early_stopping.stop:
            print("Early stopping triggered")
            break

    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "models/unet_last.pth")
    print("Saved: models/best_unet.pth and models/unet_last.pth")


if __name__ == "__main__":
    main()

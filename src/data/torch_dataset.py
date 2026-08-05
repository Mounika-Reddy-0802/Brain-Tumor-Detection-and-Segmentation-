from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import generate_pseudo_mask, load_image


class SegmentationDataset(Dataset):
    """PyTorch Dataset for segmentation that returns image/mask tensors.

    Masks come from mask_paths when given (precompute them with
    scripts/generate_masks.py). Otherwise a pseudo-mask is derived on the fly,
    which repeats the same OpenCV work every epoch.
    """

    def __init__(
        self,
        image_paths: list[str],
        mask_paths: list[str] | None = None,
        use_pseudo_masks: bool = True,
        transform=None,
        img_size: int = 224,
        preprocessed: bool = False,
    ) -> None:
        if mask_paths is None and not use_pseudo_masks:
            raise ValueError("mask_paths is required when use_pseudo_masks is False")
        if mask_paths is not None and len(mask_paths) != len(image_paths):
            raise ValueError("mask_paths and image_paths must be the same length")

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.use_pseudo_masks = use_pseudo_masks
        self.transform = transform
        self.img_size = img_size
        self.preprocessed = preprocessed

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_mask(self, idx: int) -> np.ndarray:
        if self.mask_paths is None:
            return generate_pseudo_mask(self.image_paths[idx], self.img_size)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(
                f"Failed to read mask: {self.mask_paths[idx]}. "
                "Run `python -m scripts.generate_masks --config <config>` first."
            )
        if mask.shape != (self.img_size, self.img_size):
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        return mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = load_image(self.image_paths[idx], self.img_size, self.preprocessed)
        img_uint8 = (img * 255).astype(np.uint8)
        mask = (self._load_mask(idx) > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=img_uint8, mask=mask)
            img_uint8 = augmented["image"]
            mask = augmented["mask"]

        # smp's EfficientNet encoder has no built-in preprocessing, so the
        # ImageNet normalization has to happen here.
        img_tensor = torch.FloatTensor(img_uint8).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        mask_tensor = torch.FloatTensor(mask).unsqueeze(0)
        return img_tensor, mask_tensor

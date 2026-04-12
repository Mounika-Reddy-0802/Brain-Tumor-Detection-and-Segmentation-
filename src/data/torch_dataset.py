from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import generate_pseudo_mask, preprocess_image


class SegmentationDataset(Dataset):
    """PyTorch Dataset for segmentation that returns image/mask tensors."""

    def __init__(
        self,
        image_paths: list[str],
        mask_paths: list[str] | None = None,
        use_pseudo_masks: bool = True,
        transform=None,
        img_size: int = 224,
    ) -> None:
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.use_pseudo_masks = use_pseudo_masks
        self.transform = transform
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = preprocess_image(self.image_paths[idx], self.img_size)
        img_uint8 = (img * 255).astype(np.uint8)

        if self.use_pseudo_masks or self.mask_paths is None:
            mask = generate_pseudo_mask(self.image_paths[idx], self.img_size)
        else:
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to read mask: {self.mask_paths[idx]}")
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=img_uint8, mask=mask)
            img_uint8 = augmented["image"]
            mask = augmented["mask"]

        img_tensor = torch.FloatTensor(img_uint8).permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        mask_tensor = torch.FloatTensor(mask).unsqueeze(0)
        return img_tensor, mask_tensor

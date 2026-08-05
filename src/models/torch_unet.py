from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def get_unet(encoder_name: str = "efficientnet_b3", pretrained: bool = True) -> nn.Module:
    """U-Net with EfficientNet encoder using segmentation-models-pytorch."""
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
        activation=None,
    )

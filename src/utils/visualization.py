from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Overlay a binary mask on an RGB image in red color."""
    image_u8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    mask_u8 = (mask > 0).astype(np.uint8)
    red = np.zeros_like(image_u8)
    red[mask_u8 > 0] = [255, 0, 0]
    return cv2.addWeighted(image_u8, 1.0 - alpha, red, alpha, 0)


def plot_training_curves(history: dict[str, Iterable[float]], out_path: str | Path) -> None:
    """Save line plots for all metrics present in a Keras-like history dictionary."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = [k for k in history.keys() if not k.startswith("val_")]
    n = len(keys)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        ax.plot(history.get(key, []), label=key)
        val_key = f"val_{key}"
        if val_key in history:
            ax.plot(history[val_key], label=val_key)
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

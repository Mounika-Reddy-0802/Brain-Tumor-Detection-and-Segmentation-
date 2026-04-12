from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorflow as tf
import torch


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_tf_model(model: tf.keras.Model, path: str | Path) -> None:
    ensure_parent_dir(path)
    model.save(path)


def load_tf_model(path: str | Path) -> tf.keras.Model:
    return tf.keras.models.load_model(path)


def save_torch_model(model: torch.nn.Module, path: str | Path) -> None:
    ensure_parent_dir(path)
    torch.save(model.state_dict(), path)


def load_torch_model(model: torch.nn.Module, path: str | Path, device: str | torch.device = "cpu") -> torch.nn.Module:
    state: dict[str, Any] = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model

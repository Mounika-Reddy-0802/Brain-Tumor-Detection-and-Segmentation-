from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def detection_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Binary detection metrics with recall/sensitivity as primary metric."""
    y_pred = (y_pred_probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_pred_probs)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """Multi-class classification metrics including per-class values."""
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_precision": {
            name: float(value) for name, value in zip(class_names, per_class_precision)
        },
        "per_class_recall": {
            name: float(value) for name, value in zip(class_names, per_class_recall)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def dice_coefficient(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    pred = (pred_mask > 0.5).float().view(-1)
    true = true_mask.float().view(-1)
    intersection = (pred * true).sum()
    return float((2 * intersection + smooth) / (pred.sum() + true.sum() + smooth))


def iou_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    pred = (pred_mask > 0.5).float().view(-1)
    true = true_mask.float().view(-1)
    intersection = (pred * true).sum()
    union = pred.sum() + true.sum() - intersection
    return float((intersection + smooth) / (union + smooth))

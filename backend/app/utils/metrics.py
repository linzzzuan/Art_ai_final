"""Evaluation metrics: precision, recall, F1, confusion matrix."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix as sk_confusion_matrix,
)

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Compute per-class and averaged precision/recall/F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(7)), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics = {}
    for i, name in enumerate(EMOTION_CLASSES):
        metrics[name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
        }

    return {
        "metrics": metrics,
        "macro_avg": {
            "precision": round(float(macro_p), 4),
            "recall": round(float(macro_r), 4),
            "f1": round(float(macro_f1), 4),
        },
        "weighted_avg": {
            "precision": round(float(weighted_p), 4),
            "recall": round(float(weighted_r), 4),
            "f1": round(float(weighted_f1), 4),
        },
    }


def compute_confusion_matrix(y_true: list[int], y_pred: list[int]) -> dict:
    """Compute confusion matrix and return as JSON-serializable dict."""
    cm = sk_confusion_matrix(y_true, y_pred, labels=list(range(7)))
    return {
        "labels": EMOTION_CLASSES,
        "matrix": cm.tolist(),
    }

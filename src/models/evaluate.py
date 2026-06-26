from __future__ import annotations

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect probabilities and targets from a DataLoader.
    """

    model.eval()

    all_probs: list[float] = []
    all_targets: list[float] = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy().ravel())
            all_targets.extend(targets.cpu().numpy().ravel())

    return np.array(all_probs), np.array(all_targets)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Evaluate the model using binary, macro, and weighted metrics.
    """

    model.eval()

    total_loss = 0.0
    all_probs: list[float] = []
    all_preds: list[float] = []
    all_targets: list[float] = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            loss = loss_fn(logits, targets)

            total_loss += loss.item() * features.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            all_probs.extend(probs.cpu().numpy().ravel())
            all_preds.extend(preds.cpu().numpy().ravel())
            all_targets.extend(targets.cpu().numpy().ravel())

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision": precision_score(all_targets, all_preds, zero_division=0),
        "recall": recall_score(all_targets, all_preds, zero_division=0),
        "binary_f1": f1_score(all_targets, all_preds, average="binary", zero_division=0),
        "macro_f1": f1_score(all_targets, all_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_targets, all_preds, average="weighted", zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(all_targets, all_probs)
    except ValueError:
        metrics["roc_auc"] = 0.0

    try:
        metrics["pr_auc"] = average_precision_score(all_targets, all_probs)
    except ValueError:
        metrics["pr_auc"] = 0.0

    return metrics


def tune_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    start: float = 0.10,
    stop: float = 0.90,
    step: float = 0.01,
) -> tuple[float, float]:
    """
    Find the best threshold based on validation binary F1.
    """

    best_threshold = 0.5
    best_f1 = 0.0

    thresholds = np.arange(start, stop + step, step)

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        score = f1_score(targets, preds, average="binary", zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def get_classification_report(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> str:
    """
    Return a text classification report.
    """

    preds = (probs >= threshold).astype(int)

    return classification_report(
        targets,
        preds,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    )


def get_confusion_matrix(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Return confusion matrix.
    """

    preds = (probs >= threshold).astype(int)
    return confusion_matrix(targets, preds)
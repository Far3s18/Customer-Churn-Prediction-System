from __future__ import annotations

import torch

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Train the model for one epoch.
    """

    model.train()

    total_loss = 0.0
    all_preds: list[float] = []
    all_targets: list[float] = []

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(features)
        loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        all_preds.extend(preds.detach().cpu().numpy().ravel())
        all_targets.extend(targets.detach().cpu().numpy().ravel())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision": precision_score(all_targets, all_preds, zero_division=0),
        "recall": recall_score(all_targets, all_preds, zero_division=0),
        "binary_f1": f1_score(all_targets, all_preds, average="binary", zero_division=0),
        "macro_f1": f1_score(all_targets, all_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_targets, all_preds, average="weighted", zero_division=0),
    }
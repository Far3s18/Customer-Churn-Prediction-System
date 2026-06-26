from __future__ import annotations

import torch
from torch import nn


class ANNModel(nn.Module):
    """
    Artificial Neural Network for binary tabular classification.

    The model outputs raw logits, not probabilities.
    Use BCEWithLogitsLoss during training.
    Apply sigmoid only during evaluation or inference.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropouts: list[float] | None = None,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 64, 32]

        if dropouts is None:
            dropouts = [0.1, 0.1, 0.1, 0.2, 0.1]

        if len(hidden_dims) != len(dropouts):
            raise ValueError("hidden_dims and dropouts must have the same length.")

        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim, dropout in zip(hidden_dims, dropouts):
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
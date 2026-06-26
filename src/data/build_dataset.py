from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    PyTorch Dataset for tabular binary classification.
    """

    def __init__(self, features: pd.DataFrame, targets: pd.Series | pd.DataFrame) -> None:
        self.features = torch.tensor(features.to_numpy(dtype=np.float32), dtype=torch.float32)
        self.targets = torch.tensor(targets.to_numpy(dtype=np.float32), dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]
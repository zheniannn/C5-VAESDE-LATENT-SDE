from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Returns the full (T, n_features) window — the latent SDE encodes and
    reconstructs the entire trajectory (no input/target shift like C3/C4)."""

    def __init__(self, data: np.ndarray, max_samples: Optional[int] = None, seed: int = 42) -> None:
        if data.ndim != 3 or data.shape[1:] != (30, 4):
            raise ValueError(f"Expected shape (N, 30, 4), got {data.shape}")
        if max_samples is not None and max_samples < len(data):
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(data), max_samples, replace=False))
            self._data = data[idx]
        else:
            self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self._data[idx]
        seq = seq.astype(np.float32) if isinstance(seq, np.ndarray) else np.array(seq, dtype=np.float32)
        return torch.from_numpy(seq)            # (T, n_features)

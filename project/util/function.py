import numpy as np
import torch


def one_hot_encode(label: np.ndarray) -> torch.Tensor:
    return torch.zeros(12, dtype=torch.float).scatter_(0, torch.tensor(label), value=1)
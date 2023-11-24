import numpy as np
import torch


def one_hot_encode(label: np.ndarray) -> torch.Tensor:
    return torch.zeros(12, dtype=torch.float).scatter_(0, torch.tensor(label), value=1)


def make_weights(class_length: np.ndarray) -> list:
    total_length = np.sum(class_length)
    class_weights = total_length / class_length

    weights = []
    for i, length in enumerate(class_length):
        weights += [class_weights[i]] * length
    return weights

import torch
from torch import Tensor


def relu(x: Tensor):
    return torch.maximum(x, torch.zeros_like(x))

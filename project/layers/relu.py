import torch
from torch import Tensor


class Relu:
    def __init__(self):
        self.x = None
        self.x_check = None

    def forward(self, x: Tensor):
        relu_result = torch.maximum(x, torch.zeros_like(x))
        self.x_check = (relu_result > 0).float()
        return relu_result

    def backward(self, dout: Tensor):
        return dout * self.x_check

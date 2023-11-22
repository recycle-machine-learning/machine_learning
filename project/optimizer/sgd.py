import torch

from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, lr)

    def step(self):
        for param in self.params:
            with torch.autograd.no_grad():
                param.data -= param.grad * self.lr

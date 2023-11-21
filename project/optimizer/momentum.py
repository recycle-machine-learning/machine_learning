import torch

from .optimizer import Optimizer


class Momentum(Optimizer):
    def __init__(self, params, lr, momentum=0.9):
        self.v = None
        self.momentum = momentum
        super().__init__(params, lr)

    def init_params(self, params):
        super().init_params(params)
        self.v = []
        for param in self.params:
            self.v.append(torch.zeros_like(param))

    def step(self):
        for i, param in enumerate(self.params):
            self.v[i] = self.momentum * self.v[i] - self.lr * param.grad
            param.data += self.v[i]

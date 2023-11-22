import torch

from .optimizer import Optimizer


class Momentum(Optimizer):
    def __init__(self, params, lr, momentum=0.9):
        self.v = None
        self.momentum = momentum
        super().__init__(params, lr)

    def init_params(self, params):
        super().init_params(params)
        self.v = [torch.zeros_like(param) for param in self.params]

    def step(self):
        with torch.autograd.no_grad():
            for i, param in enumerate(self.params):
                self.v[i] = self.momentum * self.v[i] - self.lr * param.grad
                param.data += self.v[i]

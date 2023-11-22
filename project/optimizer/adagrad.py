import torch

from .optimizer import Optimizer


class Adagrad(Optimizer):
    def __init__(self, params, lr):
        self.h = None
        super().__init__(params, lr)

    def init_params(self, params):
        super().init_params(params)
        self.h = [torch.zeros_like(param) for param in self.params]

    def step(self):
        with torch.autograd.no_grad():
            for i, param in enumerate(self.params):
                self.h[i] += param.grad * param.grad
                param.data -= self.lr * param.grad / (torch.sqrt(self.h[i]) + 1e-15)

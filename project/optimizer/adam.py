import torch

from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, params, lr, beta=(0.9, 0.999)):
        self.h = None
        self.beta1, self.beta2 = beta
        self.m = None
        self.v = None
        super().__init__(params, lr)

    def init_params(self, params):
        super().init_params(params)
        self.m = []
        self.v = []
        for param in self.params:
            self.m.append(torch.zeros_like(param))
            self.v.append(torch.zeros_like(param))

    def step(self):
        for i, param in enumerate(self.params):
            self.m[i] = self.m[i] * self.beta1 + (1 - self.beta1) * param.grad
            self.v[i] = self.v[i] * self.beta2 + (1 - self.beta2) * param.grad * param.grad
            param.data -= self.lr * self.m[i] / (torch.sqrt(self.v[i]) + 1e-15)

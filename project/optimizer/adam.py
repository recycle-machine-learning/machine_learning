import torch

from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, params, lr, beta=(0.9, 0.999)):
        self.beta1, self.beta2 = beta
        self.m = None
        self.v = None
        super().__init__(params, lr)

    def init_params(self, params):
        super().init_params(params)
        self.m = [torch.zeros_like(param) for param in self.params]
        self.v = [torch.zeros_like(param) for param in self.params]

    def step(self):
        with torch.autograd.no_grad():
            for i, param in enumerate(self.params):
                self.m[i] = self.m[i] * self.beta1 + (1 - self.beta1) * param.grad
                self.v[i] = self.v[i] * self.beta2 + (1 - self.beta2) * param.grad * param.grad
                param.data -= self.lr * self.m[i] / (torch.sqrt(self.v[i]) + 1e-15)

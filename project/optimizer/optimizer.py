import torch


class Optimizer:
    def __init__(self, params, lr):
        self.lr = lr

        self.params = None
        self.init_params(params)

    def init_params(self, params):
        self.params = [param for param in params]

    def zero_grad(self):
        with torch.no_grad():
            for param in self.params:
                param.grad = None

    def step(self):
        pass

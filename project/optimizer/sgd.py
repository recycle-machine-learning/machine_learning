from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, lr)

    def step(self):
        for param in self.params:
            param.data = param.data - param.grad * self.lr

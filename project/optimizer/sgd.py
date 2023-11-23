import torch

from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, lr)

    def step(self):
        """
        mini_batch 단위로 호출, 합산된 변화량으로 Parameter 갱신
        """
        for param in self.params:
            with torch.autograd.no_grad():
                param.data -= param.grad * self.lr

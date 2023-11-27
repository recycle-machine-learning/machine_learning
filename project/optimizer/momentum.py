import torch

from .optimizer import Optimizer


class Momentum(Optimizer):
    def __init__(self, params, lr, momentum=0.9):
        self.v = None
        self.momentum = momentum
        super().__init__(params, lr)

    def init_params(self, params) -> None:
        super().init_params(params)
        # 각 Parameter와 같은 형상의 텐서를 0으로 초기화
        self.v = [torch.zeros_like(param) for param in self.params]

    def step(self) -> None:
        for i, param in enumerate(self.params):
            # 가속도 개념 도입
            self.v[i] = self.momentum * self.v[i] - self.lr * param.grad
            param.data += self.v[i]

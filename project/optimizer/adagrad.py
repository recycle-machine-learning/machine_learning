import torch

from .optimizer import Optimizer


class Adagrad(Optimizer):
    def __init__(self, params, lr):
        self.h = None
        super().__init__(params, lr)

    def init_params(self, params) -> None:
        super().init_params(params)
        # 각 Parameter와 같은 형상의 텐서를 0으로 초기화
        self.h = [torch.zeros_like(param) for param in self.params]

    def step(self) -> None:
        for i, param in enumerate(self.params):
            # 이전까지 더 많이 변화했을수록 더 적게 변화
            self.h[i] += param.grad * param.grad
            param.data -= self.lr * param.grad / (torch.sqrt(self.h[i]) + 1e-15)

import torch

from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Momentum, Adagrad를 혼합한 최적화 기법
    """
    def __init__(self, params, lr, beta=(0.9, 0.999)):
        """
        :param params: model의 모든 Parameter의 iterator
        :param lr: 학습률
        :param beta: momentum 계수
        """
        self.beta1, self.beta2 = beta
        self.m = None
        self.v = None
        # Parameter, lr 초기화
        super().__init__(params, lr)

    def init_params(self, params) -> None:
        super().init_params(params)
        # 각 Parameter와 같은 형상의 텐서를 0으로 초기화
        self.m = [torch.zeros_like(param, requires_grad=False) for param in self.params]
        self.v = [torch.zeros_like(param, requires_grad=False) for param in self.params]

    def step(self, c=1e-15) -> None:
        for i, param in enumerate(self.params):
            # Momentum
            self.m[i] = self.m[i] * self.beta1 + (1 - self.beta1) * param.grad
            # Adagrad
            self.v[i] = self.v[i] * self.beta2 + (1 - self.beta2) * param.grad * param.grad
            param.data -= self.lr * self.m[i] / (torch.sqrt(self.v[i]) + c)

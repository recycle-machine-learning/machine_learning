import torch

from .optimizer import Optimizer


class RMSprop(Optimizer):
    def __init__(self, params, lr, weight_decay=0.99):
        """

        :param params:
        :param lr:
        :param weight_decay:
        """
        self.h = None
        self.weight_decay = weight_decay
        super().__init__(params, lr)

    def init_params(self, params) -> None:
        super().init_params(params)
        # 각 Parameter와 같은 형상의 텐서를 0으로 초기화
        self.h = [torch.zeros_like(param) for param in self.params]

    def step(self) -> None:
        with torch.autograd.no_grad():
            for i, param in enumerate(self.params):
                # Adagrad에서 오래된 기울기의 영향이 점점 사라짐
                self.h[i] = self.h[i] * self.weight_decay + (1 - self.weight_decay) * param.grad * param.grad
                param.data -= self.lr * param.grad / (torch.sqrt(self.h[i]) + 1e-15)

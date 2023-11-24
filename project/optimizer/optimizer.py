import torch


class Optimizer:
    """
    모든 Optimizer의 base Class
    """
    def __init__(self, params, lr):
        """
        :param params: model의 모든 Parameter의 iterator
        :param lr: 학습률
        """
        self.lr = lr
        self.params = None
        self.init_params(params)

    def init_params(self, params) -> None:
        """
        iterator를 받아 멤버 리스트로 초기화
        :param params: model의 모든 Parameter의 iterator
        """
        self.params = [param for param in params]

    def zero_grad(self) -> None:
        """
        모든 parameter의 grad를 None으로 초기화
        """
        with torch.no_grad():
            for param in self.params:
                param.grad = None

    def step(self) -> None:
        """
        상속 시 오버라이딩 필요
        """
        pass

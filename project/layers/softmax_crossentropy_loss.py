import torch
from torch import Tensor


class SoftmaxCrossEntropyLoss:
    def __init__(self):
        """
        :param x: Tensor(batch_size, class_size),
        :param y: Tensor(batch_size, class_size)
        :param loss: Tensor(1), mini_batch 교차 엔트로피 손실
        """
        self.loss = None
        self.x = None  # softmax 결과 값
        self.y = None  # 정답 label
        self.batch = None


    def __call__(self, x: Tensor, y: Tensor) -> Tensor:  # 순전파
        """
        입력값에 softmax 적용후 교차 엔트로피 손실 값 계산
        """
        self.x = self.softmax(x)
        self.y = y
        self.loss = self.cross_entropy(self.x, self.y)
        return self.loss

    def softmax(self, x: Tensor) -> Tensor:
        """
        0과 1 사이 값으로 정규화
        :return: Tensor(batch_size, class_size)
        """
        max_vals, _ = torch.max(x, dim=1, keepdim=True) # overflow 방지를 위한 값
        exp_x = torch.exp(x - max_vals)
        return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

    def cross_entropy(self, x: Tensor, y: Tensor, c=1e-15) -> Tensor:
        """
        mini_batch의 평균 교차 엔트로피 손실 계산
        :param x: Tensor(batch_size, class_size), softmax 연산 결과
        :param y: Tensor(batch_size, class_Size), one-hot-encoded 정답 label
        :param c: 오버플로 방지를 위해 더하는 작은 값
        :param batch: batch size
        :return: Tensor(1), mini_batch 교차 엔트로피 손실
        """
        self.batch = self.x.shape[0]
        return -torch.sum(y * torch.log(x + c)) / self.batch

    def backward(self):
        """
        역전파
        :param dx: Tensor(batch_size, class_size)
        """
        dx = (self.x - self.y) / self.batch  # dx = softmax - label
        return dx

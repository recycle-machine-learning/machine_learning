import torch
from torch import Tensor


class SoftmaxCrossEntropyLoss():

    def __init__(self):
        self.loss = None
        self.x = None   # softmax 결과 값
        self.y = None   # 정답

    def __call__(self, x, y):  # 순전파
        self.x = self.softmax(x)
        self.y = y
        self.loss = self.cross_entropy(self.x, self.y)
        return self.loss

    def softmax(self, x: Tensor):
        max_vals, _ = torch.max(x, dim=1, keepdim=True)
        exp_x = torch.exp(x - max_vals)
        return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

    def cross_entropy(self, x, y, c=1e-15):
        self.batch = self.x.shape[0]
        return -torch.sum(y * torch.log(x + c)) / self.batch

    def backward(self, dout=1):   # 역전파
        dx = (self.x-self.y) / self.batch  # dx = softmax - label
        return dx

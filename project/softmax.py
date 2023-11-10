import torch
from torch import Tensor



class softmax:

    def __init__(self):
        self.loss = None
        self.x = None
        self.y = None

    def softmax(self, x: Tensor):
        a = torch.max(x)
        exp_x = torch.exp(x - a)
        sum_exp_x = torch.sum(exp_x)
        y = exp_x / sum_exp_x
        return y

    def s_cross_entropy(self, x, y):
        cross_entropy = -torch.sum(torch.multiply(y, torch.log(x)))
        return cross_entropy

    def s_forward(self, x, y):
        self.x = softmax(x)
        self.y = y
        self.loss = self.s_cross_entropy(x, y)
        return self.loss

    def s_backward(self, x, y):
        return


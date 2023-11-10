import torch
from torch import Tensor



class softmax: #with loss

    def __init__(self):
        self.loss = None
        self.x = None # softmax 결과 값
        self.y = None # 정답

    def softmax(self, x: Tensor):
        a = torch.max(x)
        exp_x = torch.exp(x - a)
        sum_exp_x = torch.sum(exp_x)
        y = exp_x / sum_exp_x
        return y

    def s_cross_entropy(self, x, y):
        c=1e-7
        batch = x.shape[0]
        cross_entropy = -torch.sum(torch.multiply(y, torch.log(x + c))) / batch
        return cross_entropy

    def s_forward(self, x, y): #순전파
        self.x = softmax(x)
        self.y = y
        self.loss = self.s_cross_entropy(x, y)
        return self.loss

    def s_backward(self, dout=1): #역전파
        batch = self.x.shape[0]
        dx = (self.x-self.y)/batch #dx= softmax - label
        return dx
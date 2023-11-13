import torch

# x Shape = (n, 64 * 64 * 3)
# w Shape = (64 * 64 * 3, 12)
# dout = (n, 12)
class Affine:
    def __init__(self, w, b):   # w 2차원으로 받음
        self.w = w
        self.b = b
        self.x_reshape = None
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        batch_size = x.size(dim = 0)
        self.x = x
        self.x_reshape = x.view(batch_size, -1)
        print(self.x_reshape)
        out = self.x_reshape.matmul(self.w) + self.b
        print(self.x_reshape.matmul(self.w))

        return out

    def backward(self, dout):
        dx =  dout.matmul(torch.t(self.w))
        self.dw = torch.t(self.x_reshape).matmul(dout)
        self.db = torch.sum(dout, dim = 0)
        dx = dx.view(self.x.shape)

        return dx






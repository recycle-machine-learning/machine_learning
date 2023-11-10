import numpy as np
import torch
from torch.nn import Fold, Unfold


class Convolution:
    def __init__(self, w, b, stride=1, padding=0):
        self.w = w
        self.b = b
        self.stride = stride
        self.padding = padding

        self.x = None
        self.col = None
        self.col_w = None

        self.dw = None
        self.db = None

    def forward(self, x):
        fn, c, fh, fw = self.w.shape
        n, c, h, w = x.shape
        out_h = int(1 + (h + 2 * self.padding - fh) / self.stride)
        out_w = int(1 + (w + 2 * self.padding - fw) / self.stride)

        unfold = Unfold(kernel_size=fw, padding=self.padding, stride=self.stride)
        col = unfold(x).permute(0, 2, 1)
        col_w = self.w.view(fn, -1).T

        self.x = x
        self.col = col
        self.col_w = col_w

        y = torch.matmul(col, col_w) + self.b
        y = y.view(n, out_h, out_w, -1).permute(0, 3, 1, 2)
        return y

    def backward(self, dy):
        fn, c, fh, fw = self.w.shape
        dy = dy.permute(0, 2, 3, 1).view(-1, fn)

        self.db = torch.sum(dy, dim=0)
        self.dw = torch.matmul(self.col.T, dy)
        self.dw = self.dw.permute(1, 0).view(fn, c, fh, fw)

        col = torch.matmul(dy, self.col_w.T)

        fold = Fold(output_size=2, kernel_size=fw, padding=self.padding, stride=self.stride)
        dx = fold(col)
        return dx


if __name__ == '__main__':
    x_shape = (1, 1, 3, 3)
    x = torch.randint(0, 5, x_shape).type(torch.FloatTensor)
    print(x.shape)

    w_shape = (2, 1, 2, 2)
    w = torch.randint(0, 5, w_shape).type(torch.FloatTensor)
    print(w.shape)

    b = 1
    print(b)

    c = Convolution(w, b)
    c.forward(x)

    # c.backward()


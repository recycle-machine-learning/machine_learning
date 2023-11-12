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

        unfold = Unfold(kernel_size=(fh, fw), padding=self.padding, stride=self.stride)
        col = unfold(x).permute(0, 2, 1)
        col_w = self.w.view(fn, -1).T

        self.x = x
        self.col = col
        self.col_w = col_w

        y = torch.matmul(col, col_w) + self.b.permute(2, 1, 0)
        y = y.view(n, out_h, out_w, -1).permute(0, 3, 1, 2)
        return y

    def backward(self, dy):
        fn, c, fh, fw = self.w.shape
        dy = dy.permute(0, 2, 3, 1).reshape(-1, fn)

        self.db = torch.sum(dy, dim=0)

        reshape_col = self.col.reshape(-1, c * fh * fw)
        mul = torch.matmul(reshape_col.T, dy)
        self.dw = mul.permute(1, 0).view(fn, c, fh, fw)

        col = torch.matmul(dy, self.col_w.T)
        col = col.view(self.col.shape).permute(0, 2, 1)

        n, c, h, w = self.x.shape
        fold = Fold(output_size=(h, w), kernel_size=(fh, fw), padding=self.padding, stride=self.stride)
        dx = fold(col)
        return dx

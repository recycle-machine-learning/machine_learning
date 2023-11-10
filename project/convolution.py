import numpy as np
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
        col = unfold(x)
        col_w = self.w.reshape(fn, -1).T

        self.x = x
        self.col = col
        self.col_w = col_w

        out = np.dot(col, col_w) + self.b
        out = out.reshape(n, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, dy):
        fn, c, fh, fw = self.w.shape
        dy = dy.transpose(0, 2, 3, 1).reshape(-1, fn)

        self.db = np.sum(dy, axis=0)
        self.dw = np.dot(self.col.T, dy)
        self.dw = self.dw.transpose(1, 0).reshape(fn, c, fh, fw)

        col = np.dot(dy, self.col_w.T)

        fold = Fold(output_size=2, kernel_size=fw, padding=self.padding, stride=self.stride)
        dx = fold(col)
        return dx

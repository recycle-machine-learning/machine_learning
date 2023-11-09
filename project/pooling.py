import torch
from torch import Tensor, nn
import datetime

"""
kernel, stride, padding 개념 참조 (https://hi-lu.tistory.com/entry/파이썬으로-기초-CNN-구현하기-1-conv-pooling-layer)
"""


class MaxPooling:
    def __init__(self, kernel_size, stride, padding):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.x = None
        self.pooling = None
        self.max_indices = None

    def forward(self, x: Tensor):
        """
        x : (batch_size, channel, height, width)
        kernel_size, stride, padding = (a,b,c) -> height = ((height - a + 2*c)/b +1), width = ((width - a + 2*c)/b +1)

        out:    (
                batch_size,
                channel,
                ((height - kernel_size + 2*padding)/stride +1),
                ((width - kernel_size + 2*padding)/stride +1)
                )


        forward : padding -> max
        """
        self.x = x
        # 패딩 적용
        x_padded = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))

        print()

        max_value = torch.zeros(x_padded.shape[0], x_padded.shape[1],
                                (x_padded.shape[2] - self.kernel_size + 2 * self.padding) // self.stride,
                                (x_padded.shape[3] - self.kernel_size + 2 * self.padding) // self.stride)
        self.max_indices = torch.zeros_like(x)

        for i in range(max_value.shape[2]):
            for j in range(max_value.shape[3]):
                max_value[:, :, i, j] = self.extract_max_indices(x_padded, i, j)

        return max_value

    def extract_max_indices(self, x_padded: Tensor, i, j):
        max_idx1, max_idx2 = 0, 0
        max_value = 0.0
        for d1 in range(i * self.stride, i * self.stride + self.kernel_size):
            for d2 in range(j * self.stride, j * self.stride + self.kernel_size):
                if x_padded[:, :, d1, d2] > max_value:
                    max_value = x_padded[:, :, d1, d2]
                    max_idx1, max_idx2 = d1, d2
        if max_value != 0:
            self.max_indices[:, :, max_idx1 - self.padding, max_idx2 - self.padding] = 1
        return max_value

    def backward(self, dout: Tensor):
        """
        최댓값은 해당 위치값, 나머지는 0
        """
        return dout * self.max_indices

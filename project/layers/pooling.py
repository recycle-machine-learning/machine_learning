import torch
from torch import Tensor, nn
import datetime

"""
kernel, stride, padding 개념 참조 (https://hi-lu.tistory.com/entry/파이썬으로-기초-CNN-구현하기-1-conv-pooling-layer)
"""


class MaxPooling:
    def __init__(self, kernel_size, stride, padding):
        """
        :param kernel_size: 최댓값을 구하는 영역의 크기
        :param stride: 한 번에 이동하는 크기
        :param padding: 입력 데이터의 주변을 특정 값으로 채우는 것
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.x = None
        self.x_unfold = None
        self.x_unfold_permute = None
        self.x_unfold_max_values = None


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

        """
        x_unfold = self.unfold(x)
        x_unfold_permute = (
            x_unfold.view(x.shape[0], x.shape[1], self.kernel_size * self.kernel_size, -1).permute(0, 1, 3, 2))
        x_unfold_max_values = torch.max(x_unfold_permute, dim=3).values
        pooling = x_unfold_max_values.view(x.shape[0], x.shape[1], x.shape[2] // self.stride, x.shape[3] // self.stride)

        self.x = x
        self.x_unfold = x_unfold
        self.x_unfold_permute = x_unfold_permute
        self.x_unfold_max_values = x_unfold_max_values

        return pooling

    def backward(self, dout: Tensor):
        """
        dout : (batch_size, channel, height, width)
        mask_dout_fold = (batch_size, channel, height, width)
        """

        fold = nn.Fold(output_size=(self.x.shape[2],self.x.shape[3]), kernel_size=self.kernel_size, stride=self.stride,
                       padding=self.padding)
        mask = (self.x_unfold_permute == self.x_unfold_max_values.unsqueeze(-1)).float()
        dout_expand = (dout.view(self.x_unfold_permute.shape[0],self.x_unfold_permute.shape[1],self.x_unfold_permute.shape[2],1)
                       .expand(self.x_unfold_permute.shape))
        mask_dout = mask * dout_expand
        mask_dout_permute = mask_dout.permute(0, 1, 3, 2)
        mask_dout_permute_view = mask_dout_permute.view(self.x_unfold.shape[0],self.x_unfold.shape[1],self.x_unfold.shape[2])
        mask_dout_fold = fold(mask_dout_permute_view)

        return mask_dout_fold
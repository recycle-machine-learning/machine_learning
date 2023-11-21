import math

import torch
from torch.nn import Fold, Unfold
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import init


class Convolution(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 device: str = "cpu", dtype=None):
        kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = Parameter(torch.empty(weight_shape, **kwargs))
        self.bias = Parameter(torch.empty((out_channels, 1, 1), **kwargs))

        self.reset_parameters()

        self.x = None
        self.x_im2col = None
        self.w_im2col = None

        self.weight.grad = None
        self.bias.grad = None

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        입력(x)을 저장하고 합성곱 결과 반환
        :param x: torch.Tensor(batch_size: 배치 크기, channel: 채널 크기, height: 데이터 높이, width: 데이터 너비)
        :return: 순전파 결과
        """
        filter_number, channel, filter_height, filter_width = self.weight.shape
        batch_size, channel, height, width = x.shape
        out_h = int(1 + (height + 2 * self.padding - filter_height) / self.stride)
        out_w = int(1 + (width + 2 * self.padding - filter_width) / self.stride)

        unfold = Unfold(kernel_size=(filter_height, filter_width), padding=self.padding, stride=self.stride)
        x_im2col = unfold(x).permute(0, 2, 1)
        w_im2col = self.weight.view(filter_number, -1).T

        self.x = x
        self.x_im2col = x_im2col
        self.w_im2col = w_im2col

        out = torch.matmul(x_im2col, w_im2col) + self.bias.permute(2, 1, 0)
        out = out.view(batch_size, out_h, out_w, -1).permute(0, 3, 1, 2)
        return out

    def backward(self, dout):
        filter_number, channel, filter_height, filter_width = self.weight.shape
        dout = dout.permute(0, 2, 3, 1).reshape(-1, filter_number)

        self.bias.grad = torch.reshape(torch.sum(dout, dim=0), (torch.sum(dout, dim=0).size(0),1,1))

        reshape_col = self.x_im2col.reshape(-1, channel * filter_height * filter_width)
        mul = torch.matmul(reshape_col.T, dout)
        self.weight.grad = mul.permute(1, 0).view(filter_number, channel, filter_height, filter_width)

        col = torch.matmul(dout, self.w_im2col.T)
        col = col.view(self.x_im2col.shape).permute(0, 2, 1)

        batch_size, channel, height, width = self.x.shape
        fold = Fold(output_size=(height, width), kernel_size=(filter_height, filter_width),
                    padding=self.padding, stride=self.stride)
        dx = fold(col)
        return dx

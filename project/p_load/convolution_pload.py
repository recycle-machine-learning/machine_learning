import math

import torch
from torch import Tensor
from torch.nn import Fold, Unfold
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import init


class Convolution_pload(Module):
    """
    합성곱 연산 수행
    """
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 device="cpu",
                 dtype=None,
                 w=None, b=None):
        """
        :param in_channels: 입력 채널 수
        :param out_channels: 출력 채널 수
        :param kernel_size: 필터 길이(가로, 세로 동일)
        :param stride: 필터 사이 빈 칸 길이
        :param padding: 입력에 확장할 빈 칸 길이
        :param device: 텐서 연산이 일어나는 장치
        :param dtype: Parameter(weight, bias) 타입
        """
        kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 학습이 필요한 텐서는 Parameter로 선언해야 model.parameters()에 자동으로 추가
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = w
        self.bias = b


        self.x = None
        self.x_im2col = None
        self.w_im2col = None

        self.weight.grad = None
        self.bias.grad = None

    def reset_parameters(self) -> None:
        """
        Parameter(weight, bias) He 초깃값으로 초기화
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """
        입력(x)을 저장하고 합성곱 연산 결과를 반환
        :param x: Tensor(batch_size, in_channels, height, width)
        :return: Tensor(batch_size, out_channels, height, width)
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

    def backward(self, dout: Tensor) -> Tensor:
        """
        Parameter(weight, bias) 변화량을 저장하고 입력(x)의 변화량 반환
        :param dout: Tensor(batch_size, out_channels, height, width)
        :return: Tensor(batch_size, in_channels, height, width)
        """
        filter_number, channel, filter_height, filter_width = self.weight.shape
        dout = dout.permute(0, 2, 3, 1).reshape(-1, filter_number)

        self.bias.grad = torch.reshape(torch.sum(dout, dim=0), (torch.sum(dout, dim=0).size(0), 1, 1))

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

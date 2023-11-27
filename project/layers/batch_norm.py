import torch
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.parameter import Parameter


class BatchNormalization(Module):
    def __init__(self, num_features: int,
                 epsilon: float = 1e-5,
                 momentum: float = 0.9,
                 affine: bool = True,
                 device: str = "cpu",
                 dtype=None
                 ) -> None:
        super().__init__()
        """
        :param num_features: 입력 데이터의 채널 수
        :param epsilon: 0으로 나누는 것을 방지하기 위한 작은 값
        :param momentum: 이동평균을 구할 때 이전 값에 곱해줄 값
        :param affine: scale, shift를 사용할지 여부
        :param device: cpu or gpu
        :param dtype: float16, float32, float64
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_features = num_features
        self.epsilon = epsilon  # 0으로 나누는 것을 방지하기 위한 작은 값
        self.momentum = momentum # 이동평균을 구할 때 이전 값에 곱해줄 값
        self.affine = affine # scale, shift를 사용할지 여부
        self.input_shape = None # (batch_size, channel, height, width)

        if self.affine: # scale, shift를 사용할 경우
            self.weight = Parameter(torch.ones(num_features, **factory_kwargs))  # gamma,
            self.bias = Parameter(torch.zeros(num_features, **factory_kwargs))  # beta
            # self.reset_parameters()

        self.running_mean = None # 이동평균
        self.running_var = None # 이동분산

        self.batch_size = None # 배치 사이즈
        self.xc = None # 입력값에 평균을 뺀 값
        self.x_norm = None # 정규화된 값
        self.std = None # 표준편차

        self.weight.grad = None # gamma의 기울기
        self.bias.grad = None # beta의 기울기

        self.is_training = True # 학습 중인지 여부

    def reset_parameters(self) -> None:
        if self.affine: # scale, shift를 사용할 경우
            init.ones_(self.weight) # gamma
            init.zeros_(self.bias) # beta

    def forward(self, x):
        """
        forward는 학습 중일 때와 테스트 중일 때를 구분해야 한다.
        :param x: (batch_size, channel, height, width)
        :var xc: (batch_size, channel, height, width)
        :var x_norm: (batch_size, channel, height, width)
        :var std: (batch_size, channel, height, width)
        :var running_mean: (channel, )
        :var running_var: (channel, )
        :var weight: (channel, )
        :var bias: (channel, )
        :var out: (batch_size, channel, height, width)
        :return: (batch_size, channel, height, width)
        """
        batch_size, channel, height, width = x.shape

        self.input_shape = x.shape
        self.batch_size = batch_size

        if self.running_mean is None:
            self.running_mean = torch.zeros(channel)
            self.running_var = torch.zeros(channel)

        if self.is_training:
            mean = torch.mean(x, dim=(0, 2, 3))
            mean_view = mean.view(1, -1, 1, 1)
            xc = x - mean_view
            var = torch.mean(xc ** 2, dim=(0, 2, 3))
            var_view = var.view(1, -1, 1, 1)
            std = torch.sqrt(var_view + self.epsilon)
            x_norm = xc / std

            self.xc = xc
            self.x_norm = x_norm
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else: # 학습 중이 아닐 경우
            xc = x - self.running_mean.view(1, -1, 1, 1)
            x_norm = xc / (torch.sqrt(self.running_var.view(1, -1, 1, 1) + self.epsilon))

        x_norm = x_norm.view(*self.input_shape)

        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        out = weight * x_norm + bias

        return out

    def backward(self, dout):
        """
        :param dout: (batch_size, channel, height, width)
        :var weight: (channel, )
        :var bias: (channel, )
        :var dx_norm: (batch_size, channel, height, width)
        :var dxc: (batch_size, channel, height, width)
        :var dstd_view: (1, channel, 1, 1)
        :var dvar: (channel, )
        :var dmean_view: (1, channel, 1, 1)
        :var dx: (batch_size, channel, height, width)
        :var self.weight.grad: (channel, )
        :var self.bias.grad: (channel, )
        :return: (batch_size, channel, height, width)
        """
        batch_size = dout.shape[0]

        weight = self.weight.view((1, -1, 1, 1))

        dx_norm = dout * weight

        dxc = dx_norm / self.std
        dstd = -torch.sum((dx_norm * self.xc) / (self.std * self.std), dim=(0, 2, 3))
        dstd_view = dstd.view(1, -1, 1, 1)
        dvar = 0.5 * dstd_view / self.std
        dxc += (2.0 / batch_size) * self.xc * dvar
        dmean = torch.sum(dxc, dim=(0, 2, 3))
        dmean_view = dmean.view(1, -1, 1, 1)
        dx = dxc - dmean_view / batch_size

        self.weight.grad = torch.sum(dout * self.x_norm, dim=(0, 2, 3), dtype=torch.float32)
        self.bias.grad = dout.sum(dim=(0, 2, 3), dtype=torch.float32)

        dx = dx.reshape(*dout.shape)

        return dx

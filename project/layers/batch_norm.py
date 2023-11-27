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
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_features = num_features
        self.epsilon = epsilon  # 0으로 나누는 것을 방지하기 위한 작은 값
        self.momentum = momentum
        self.affine = affine
        self.input_shape = None

        if self.affine:
            self.weight = Parameter(torch.ones(num_features, **factory_kwargs))  # gamma
            self.bias = Parameter(torch.zeros(num_features, **factory_kwargs))  # beta
            # self.reset_parameters()

        self.running_mean = None
        self.running_var = None

        self.batch_size = None
        self.xc = None
        self.x_norm = None
        self.std = None

        self.weight.grad = None
        self.bias.grad = None

        self.is_training = True

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x):
        batch_size, channel, height, width = x.shape

        self.input_shape = x.shape
        self.batch_size = batch_size

        if self.running_mean is None:
            self.running_mean = torch.zeros(channel)
            self.running_var = torch.zeros(channel)

        if self.is_training:
            mean = torch.mean(x, dim=(0, 2, 3))
            mean_view = mean.view(1, -1, 1, 1)  # 평균 (1,)
            xc = x - mean_view  # 평균을 0으로 (batch_size, channel * height * width)
            var = torch.mean(xc ** 2, dim=(0, 2, 3))
            var_view = var.view(1, -1, 1, 1)  # 분산 (channel * height * width,)
            std = torch.sqrt(var_view + self.epsilon)  # 표준편차 (batch_size, 1)
            x_norm = xc / std  # 정규화 (batch_size, channel ,height ,width)

            self.xc = xc
            self.x_norm = x_norm
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean.view(1, -1, 1, 1)
            x_norm = xc / (torch.sqrt(self.running_var.view(1, -1, 1, 1) + self.epsilon))

        x_norm = x_norm.view(*self.input_shape)

        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        out = weight * x_norm + bias

        return out

    def backward(self, dout):
        batch_size = dout.shape[0]

        weight = self.weight.view((1, -1, 1, 1))

        dx_norm = dout * weight

        dxc = dx_norm / self.std
        dstd = -torch.sum((dx_norm * self.xc) / (self.std * self.std), dim=(0, 2, 3))
        dstd_view = dstd.view(1, -1, 1, 1)  # (channel * height * width,)
        dvar = 0.5 * dstd_view / self.std  # (channel * height * width,)
        dxc += (2.0 / batch_size) * self.xc * dvar  # (batch_size, channel * height * width)
        dmean = torch.sum(dxc, dim=(0, 2, 3))  # (channel * height * width,)
        dmean_view = dmean.view(1, -1, 1, 1)
        dx = dxc - dmean_view / batch_size  # (batch_size, channel * height * width)

        self.weight.grad = torch.sum(dout * self.x_norm, dim=(0, 2, 3), dtype=torch.float32)
        self.bias.grad = dout.sum(dim=(0, 2, 3), dtype=torch.float32)

        dx = dx.reshape(*dout.shape)

        return dx

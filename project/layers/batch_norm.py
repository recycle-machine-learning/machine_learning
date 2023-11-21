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

        self.d_weight = None
        self.d_bias = None

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x, is_training=True):
        batch_size, channel, height, width = x.shape
        x_shape = x.shape

        self.batch_size = batch_size
        x = x.reshape(self.batch_size, -1)

        if self.running_mean is None:
            self.running_mean = torch.zeros(x.shape[1])
            self.running_var = torch.zeros(x.shape[1])

        if is_training:
            mean = torch.mean(x)  # 평균 (1,)
            xc = x - mean  # 평균을 0으로 (batch_size, channel * height * width)
            var = torch.mean(xc ** 2)  # 분산 (1,)
            std = torch.sqrt(var + self.epsilon)  # 표준편차 (batch_size, 1)
            x_norm = xc / std  # 정규화 (batch_size, channel * height * width)

            self.xc = xc
            self.x_norm = x_norm
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            x_norm = xc / (torch.sqrt(self.running_var + self.epsilon))

        x_norm = x_norm.reshape(x_shape)

        weight = self.weight.reshape((1, self.num_features, 1, 1))
        bias = self.bias.reshape((1, self.num_features, 1, 1))

        out = weight * x_norm + bias

        return out

    def backward(self, dout):
        dout = dout.reshape(self.batch_size, -1)

        dx_norm = dout * self.weight
        dxc = dx_norm / self.std
        dstd = -torch.sum((dx_norm * self.xc) / (self.std * self.std), dim=1, keepdim=True)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmean = torch.sum(dxc, dim=1, keepdim=True)
        dx = dxc - dmean / self.batch_size

        dx = dx.reshape(self.batch_size, *self.x_shape[1:])
        self.d_weight = torch.sum(dout * self.x_norm, dim=(0, 2, 3))
        self.d_bias = torch.sum(dout, dim=(0, 2, 3))

        return dx

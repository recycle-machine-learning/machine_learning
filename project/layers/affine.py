import math

import torch
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import init


# x Shape = (batch_size, channel, width, height)
# x_reshape = (batch_size, input_features)
# w Shape = (channel * width * height, output_features)
# dout = (batch_size, output_features)
class Affine(Module):
    def __init__(self, input_features, output_features, bias, device='cpu', dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.weight = Parameter(torch.empty((output_features, input_features), **factory_kwargs))
        if bias:
            self.b = Parameter(torch.empty(output_features, **factory_kwargs))
        else:
            self.register_parameter('b', None)

        self.bias = bias

        # 가중치 초기화
        init.kaiming_normal_(self.weight, a=math.sqrt(5), nonlinearity='relu')
        if self.bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.b, -bound, bound)

        self.x_reshape = None
        self.x = None
        self.dw = None
        self.db = None

    """
    # input_features = channel * width * height
    # x.shape = (batch_size, channel, width, height)
    # x_reshape.shape = (batch_size, input_features) 
    # weight.shape = (output_features, input_features)
    # out.shape = (batch_size, output_features) 
    """
    def forward(self, x):
        self.x = x
        out = (self.x.matmul(self.weight.T))
        if self.bias:
            out += self.b

        return out

    def backward(self, dout):
        dx = dout.matmul(self.weight)
        self.dw = dout.T.matmul(self.x_reshape)
        self.db = torch.sum(dout, dim=0)
        dx = dx.view(self.x.shape)
        return dx

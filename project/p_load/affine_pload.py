import math

import torch
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import init


# x Shape = (batch_size, channel, width, height)
# x_reshape = (batch_size, input_features)
# w Shape = (channel * width * height, output_features)
# dout = (batch_size, output_features)
class Affine_pload(Module):
    def __init__(self, input_features, output_features, device='cpu', dtype=None, w=None, b=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.weight = w
        self.b = b

        self.x_reshape = None
        self.x = None
        self.weight.grad = None
        self.b.grad = None

    """
    # input_features = channel * width * height
    # x.shape = (batch_size, channel, width, height)
    # x_reshape.shape = (batch_size, input_features) 
    # weight.shape = (output_features, input_features)
    # out.shape = (batch_size, output_features) 
    """
    def forward(self, x):
        # out = out.reshape(out.size(0), -1)

        self.x = x
        self.x_reshape = x.reshape(x.size(0), -1)
        out = (self.x_reshape.matmul(self.weight.T))
        out += self.b

        return out

    def backward(self, dout):
        dx = dout.matmul(self.weight)
        self.weight.grad = dout.T.matmul(self.x_reshape)
        self.b.grad = torch.sum(dout, dim=0)
        dx = dx.view(self.x.shape)
        return dx

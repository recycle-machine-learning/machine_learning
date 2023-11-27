import math

import torch
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import init

"""
:param input_features: 입력 features 수
:param out_features: 출력 features 수
:param device: 텐서 연산이 일어나는 장치
:param dtype: Parameter(weight, bias) 타입
:param bias: bias 값을 적용할지에 대한 bool 값
:param x_reshape: 행렬 곱을 하기 위해 2차원으로 변환한 x
:param x: x 값
:param weight.grad: weight 값
:parma b.grad: bias 값 
"""
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
        self.weight.grad = None
        self.b.grad = None

    """
    # input_features = channel * width * height
    # x.shape: (batch_size, channel, width, height)
    # x_reshape.shape: (batch_size, input_features) 
    # weight.shape: (output_features, input_features)
    # out.shape: (batch_size, output_features) 
    """
    def forward(self, x):
        # 4차원 형태를 저장
        self.x = x
        # 2차원 형태로 변환
        self.x_reshape = x.reshape(x.size(0), -1)
        out = self.x_reshape.matmul(self.weight.T)
        if self.bias:
            out += self.b

        return out

    """
     # dx.shape: (batch_size, channel * width * height)
     # weight.grad.shape: (output_features, input_features)
     # out.shape: (batch_size, output_features) 
     """
    def backward(self, dout):
        dx = dout.matmul(self.weight)
        self.weight.grad = dout.T.matmul(self.x_reshape)
        self.b.grad = torch.sum(dout, dim=0)
        # 처음 들어왔던 4차원으로 변경
        dx = dx.view(self.x.shape)
        return dx

import torch

# x Shape = (batch_size, channel, width, height)
# x_reshape = (batch_size, input_features)
# w Shape = (channel * width * height, output_features)
# dout = (batch_size, output_features)
class Affine:
    def __init__(self, input_features, output_features, bias):
        self.weight = torch.randn((output_features, input_features), dtype=torch.float32, device="mps")
        self.b = torch.randn(output_features, dtype=torch.float32, device="mps")
        self.bias = bias
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
        batch_size = x.size(dim = 0)
        self.x = x
        self.x_reshape = x.view(batch_size, -1)
        out = (self.x_reshape.matmul(self.weight.T))
        if self.bias:
            out += self.b

        return out

    def backward(self, dout):
        dx =  dout.matmul(self.weight)
        self.dw = dout.T.matmul(self.x_reshape)
        self.db = torch.sum(dout, dim = 0)
        dx = dx.view(self.x.shape)
        return dx






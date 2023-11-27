import torch.nn as nn
from project.p_load.model_parameter import *
from layers import *
from p_load import convolution_pload
from p_load import affine_pload

class CNN_pload(nn.Module):
    def __init__(self, size, out_channel1, out_channel2):
        super(CNN_pload, self).__init__()
        # 첫 번째 층
        # Image Shape = (?, 128, 128, 3)
        # CONV = (?, 128, 128, 16)
        # POOL = (?, 64, 64, 16)
        p = model_parameter()
        conv1weight = p.load_parameters(p_name = "conv1weight",shape = (out_channel1,3, 3, 3))
        conv1bias = p.load_parameters(p_name = "conv1bias",shape = (out_channel1,1,1))
        self.conv1 = convolution_pload.Convolution_pload(in_channels=3, out_channels=out_channel1,
                                                         kernel_size=3, stride=1, padding=1, w=conv1weight, b=conv1bias)
        self.bn1 = BatchNormalization(out_channel1)
        self.relu1 = Relu()
        self.pool1 = MaxPooling(kernel_size=2, stride=2, padding=0)



        # 두 번째 층
        # Image Shape = (?, 64, 64, 16)
        # CONV = (?, 64, 64, 32)
        # POOL = (?, 32, 32, 32)
        conv2weight = p.load_parameters(p_name="conv2weight", shape =(out_channel2, out_channel1, 3, 3))
        conv2bias = p.load_parameters(p_name="conv2bias", shape=(out_channel2, 1, 1))
        self.conv2 = convolution_pload.Convolution_pload(in_channels=out_channel1, out_channels=out_channel2,
                                                         kernel_size=3, stride=1, padding=1, w=conv2weight, b=conv2bias)
        self.bn2 = BatchNormalization(out_channel2)
        self.pool2 = MaxPooling(kernel_size=2, stride=2, padding=0)
        self.relu2 = Relu()
        fc1weight = p.load_parameters(p_name="fc1weight", shape =(12,(size // 4) * (size // 4) * out_channel2))
        fc1b = p.load_parameters(p_name="fc1b", shape=(12))
        self.fc1 = affine_pload.Affine_pload((size // 4) * (size // 4) * out_channel2, 12, w=fc1weight, b=fc1b)


    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)

        out = self.fc1.forward(out)

        return out

import torch.nn as nn

from layers import *


class CNN(nn.Module):
    def __init__(self, size, out_channel1, out_channel2):
        super(CNN, self).__init__()
        # 첫 번째 층
        # Image Shape = (?, 128, 128, 3)
        # CONV = (?, 128, 128, 16)
        # POOL = (?, 64, 64, 16)

        self.conv1 = Convolution(in_channels=3, out_channels=out_channel1, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNormalization(out_channel1)
        self.relu1 = Relu()
        self.pool1 = MaxPooling(kernel_size=2, stride=2, padding=0)

        # 두 번째 층
        # Image Shape = (?, 64, 64, 16)
        # CONV = (?, 64, 64, 32)
        # POOL = (?, 32, 32, 32)

        self.conv2 = Convolution(in_channels=out_channel1, out_channels=out_channel2, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNormalization(out_channel2)
        self.pool2 = MaxPooling(kernel_size=2, stride=2, padding=0)
        self.relu2 = Relu()

        self.fc1 = Affine((size // 4) * (size // 4) * out_channel2, 12, bias=True)


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

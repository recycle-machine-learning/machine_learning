import torch.nn as nn

from layers import *


class CNN(nn.Module):
    def __init__(self, size, out_channel1, out_channel2):
        super(CNN, self).__init__()
        # 첫 번째 층
        # Image Shape = (?, 128, 128, 3)
        # CONV = (?, 128, 128, 16)
        # POOL = (?, 64, 64, 16)
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_test = Convolution(in_channels=3, out_channels=out_channel1, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(16)
        self.bn1_test = BatchNormalization(out_channel1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1_test = MaxPooling(kernel_size=2, stride=2, padding=0)

        # 두 번째 층
        # Image Shape = (?, 64, 64, 16)
        # CONV = (?, 64, 64, 32)
        # POOL = (?, 32, 32, 32)
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_test = Convolution(in_channels=out_channel1, out_channels=out_channel2, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(32)
        self.bn2_test = BatchNormalization(out_channel2)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2_test = MaxPooling(kernel_size=2, stride=2, padding=0)

        # self.fc1 = nn.Linear(8 * 8 * 32, 12, bias=True)
        self.fc1_test = Affine((size // 4) * (size // 4) * out_channel2, 12, bias=True)

        # self.drop1 = nn.Dropout(0.2)
        # self.drop2 = nn.Dropout(0.2)
        # self.drop3 = nn.Dropout(0.2)

        self.relu1 = Relu()
        self.relu2 = Relu()

    def forward(self, x):
        # out = self.conv1(x)
        out = self.conv1_test.forward(x)
        # out = self.bn1(out)
        out = self.bn1_test.forward(out)
        out = self.relu1.forward(out)
        # out = self.drop1(out)
        # out = self.pool1(out)
        out = self.pool1_test.forward(out)

        # out = self.conv2(out)
        out = self.conv2_test.forward(out)
        # out = self.bn2(out)
        out = self.bn2_test.forward(out)
        out = self.relu2.forward(out)
        # out = self.drop2(out)
        # out = self.pool2(out)
        out = self.pool2_test.forward(out)

        # out = out.reshape(out.size(0), -1)
        # out = self.fc1(out)
        out = self.fc1_test.forward(out)
        # out = self.drop3(out)
        return out

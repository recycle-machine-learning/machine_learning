import torch.nn as nn

from project.layers import relu, pooling, convolution, affine


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 첫 번째 층
        # Image Shape = (?, 128, 128, 3)
        # CONV = (?, 128, 128, 16)
        # POOL = (?, 64, 64, 16)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_test = convolution.Convolution(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1_test = pooling.MaxPooling(kernel_size=2, stride=2, padding=0)

        # 두 번째 층
        # Image Shape = (?, 64, 64, 16)
        # CONV = (?, 64, 64, 32)
        # POOL = (?, 32, 32, 32)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_test = convolution.Convolution(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2_test = pooling.MaxPooling(kernel_size=2, stride=2, padding=0)


        self.fc1 = nn.Linear(32 * 32 * 64, 12, bias=True)
        self.fc1_test = affine.Affine(32 * 32 * 64, 12, bias=True)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.25)
        self.drop3 = nn.Dropout(0.25)

        self.relu = relu.Relu()

    def forward(self, x):
        # out = self.conv1(x)
        out = self.conv1_test.forward(x)
        out = self.bn1(out)
        out = self.relu.forward(out)
        out = self.drop1(out)
        # out = self.pool1(out)
        out = self.pool1_test.forward(out)

        # out = self.conv2(out)
        out = self.conv2_test.forward(out)
        out = self.bn2(out)
        out = self.relu.forward(out)
        out = self.drop2(out)
        # out = self.pool2(out)
        out = self.pool2_test.forward(out)

        out = out.reshape(out.size(0), -1)
        # out = self.fc1(out)
        out = self.fc1_test.forward(out)
        out = self.drop3(out)
        return out

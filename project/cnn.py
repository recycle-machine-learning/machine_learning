import torch
import torch.nn as nn
import torch.nn.functional as f
import relu

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 첫 번째 층
        # Image Shape = (?, 128, 128, 3)
        # CONV = (?, 128, 128, 16)
        # POOL = (?, 64, 64, 16)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1 ,padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 두 번째 층
        # Image Shape = (?, 64, 64, 16)
        # CONV = (?, 64, 64, 32)
        # POOL = (?, 32, 32, 32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 16 * 64, 12, bias=True)

        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = relu.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = relu.relu(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out
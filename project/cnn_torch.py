import torch.nn as nn

class CNNTorch(nn.Module):
    def __init__(self, size, out_channel1, out_channel2, label_size):
        super(CNNTorch, self).__init__()

        # 첫 번째 층
        # Image Shape = (?, 128, 128, 3)
        # CONV = (?, 128, 128, 16)
        # POOL = (?, 64, 64, 16)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channel1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        # 두 번째 층
        # Image Shape = (?, 64, 64, 16)
        # CONV = (?, 64, 64, 32)
        # POOL = (?, 32, 32, 32)

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel1, out_channels=out_channel2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Linear((size // 4) * (size // 4) * out_channel2, label_size, bias=True)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer3(out)

        return out

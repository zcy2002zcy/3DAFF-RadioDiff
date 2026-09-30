import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResStage(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, blocks=2):
        super().__init__()
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride=1),
            ResidualBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.conv(x)


class ResNetBaseline(nn.Module):
    """ResNet-style encoder-decoder for dense radio-map regression.

    This is not a U-Net: decoder stages do not concatenate encoder skip maps.
    Residual blocks provide the main inductive bias.
    """

    def __init__(self, in_channels=3, out_channels=1, base_channels=64, blocks_per_stage=2):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.enc1 = ResStage(c, c, stride=1, blocks=blocks_per_stage)
        self.enc2 = ResStage(c, c * 2, stride=2, blocks=blocks_per_stage)
        self.enc3 = ResStage(c * 2, c * 4, stride=2, blocks=blocks_per_stage)
        self.enc4 = ResStage(c * 4, c * 8, stride=2, blocks=blocks_per_stage)
        self.bottleneck = ResStage(c * 8, c * 8, stride=1, blocks=blocks_per_stage)
        self.up3 = UpBlock(c * 8, c * 4)
        self.up2 = UpBlock(c * 4, c * 2)
        self.up1 = UpBlock(c * 2, c)
        self.refine = nn.Sequential(
            ResidualBlock(c, c),
            nn.Conv2d(c, out_channels, kernel_size=1),
        )

    def forward(self, x):
        size0 = x.shape[-2:]
        x = self.stem(x)
        x = self.enc1(x)
        size1 = x.shape[-2:]
        x = self.enc2(x)
        size2 = x.shape[-2:]
        x = self.enc3(x)
        size3 = x.shape[-2:]
        x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.up3(x, size3)
        x = self.up2(x, size2)
        x = self.up1(x, size1)
        x = F.interpolate(x, size=size0, mode="bilinear", align_corners=False)
        return self.refine(x)

import torch
import torch.nn as nn


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x, skip):
        return torch.cat([self.net(x), skip], dim=1)


class Pix2PixGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super().__init__()
        c = base_channels
        self.d1 = Down(in_channels, c, False)
        self.d2 = Down(c, c * 2)
        self.d3 = Down(c * 2, c * 4)
        self.d4 = Down(c * 4, c * 8)
        self.d5 = Down(c * 8, c * 8)
        self.d6 = Down(c * 8, c * 8)
        self.d7 = Down(c * 8, c * 8)
        self.b = Down(c * 8, c * 8, False)
        self.u1 = Up(c * 8, c * 8, 0.5)
        self.u2 = Up(c * 16, c * 8, 0.5)
        self.u3 = Up(c * 16, c * 8, 0.5)
        self.u4 = Up(c * 16, c * 8)
        self.u5 = Up(c * 16, c * 4)
        self.u6 = Up(c * 8, c * 2)
        self.u7 = Up(c * 4, c)
        self.out = nn.Sequential(nn.ConvTranspose2d(c * 2, out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        b = self.b(d7)
        u1 = self.u1(b, d7)
        u2 = self.u2(u1, d6)
        u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3)
        u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)
        return self.out(u7)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4, base_channels=64):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 8, 1, 4, 1, 1),
        )

    def forward(self, cond, image):
        return self.net(torch.cat([cond, image], dim=1))

# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, dim=2):
        super(double_conv, self).__init__()

        assert dim in [1, 2]

        if dim == 1:
            self.conv = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 3, padding=1),
                nn.InstanceNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_ch, out_ch, 3, padding=1),
                nn.InstanceNorm1d(out_ch),
                nn.ReLU(inplace=True)
            )

        elif dim == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dim=2):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, dim=dim)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, dim=2):
        super(down, self).__init__()

        assert dim in [1, 2]

        if dim == 1:
            self.mpconv = nn.Sequential(
                nn.MaxPool1d(2),
                double_conv(in_ch, out_ch, dim=dim)
            )

        elif dim == 2:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch, dim=dim)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, dim=2):
        super(up, self).__init__()

        assert dim in [1, 2]

        if dim == 1:
            self.up = nn.ConvTranspose1d(in_ch // 2, in_ch // 2, 2, stride=2)
        elif dim == 2:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, dim=dim)
        self.dim = dim

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if self.dim == 1:
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            x1 = F.pad(x1, (diffY // 2, diffY - diffY // 2,))

        elif self.dim == 2:
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2))

            # for padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, dim=2):
        super(outconv, self).__init__()
        if dim == 1:
            self.conv = nn.Conv1d(in_ch, out_ch, 1)
        elif dim == 2:
            self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
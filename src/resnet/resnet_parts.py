# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class residual_block(nn.Module):

    def __init__(self, channels, s_filter, dim=2):
        super(residual_block, self).__init__()

        assert dim in [1, 2]

        if dim == 1:
            self.conv = nn.Sequential(
                # nn.InstanceNorm1d(channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(channels, channels, s_filter, padding=s_filter // 2),
                # nn.InstanceNorm1d(channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(channels, channels, s_filter, padding=s_filter // 2),
            )

        elif dim == 2:
            self.conv = nn.Sequential(
                nn.InstanceNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, s_filter, padding=s_filter // 2),
                nn.InstanceNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, s_filter, padding=s_filter // 2),
            )

    def forward(self, x):
        fx = self.conv(x)
        fx.add_(x)
        return fx


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

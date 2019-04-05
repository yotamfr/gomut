# full assembly of the sub-parts to form the complete net
from .unet_parts_w_dim import *


class UNet2d(nn.Module):
    def __init__(self, n_channels, n_classes, inner_channels=64):
        super(UNet2d, self).__init__()
        self.inc = inconv(n_channels, inner_channels)
        self.down1 = down(inner_channels, inner_channels*2)
        self.down2 = down(inner_channels*2, inner_channels*4)
        self.down3 = down(inner_channels*4, inner_channels*8)
        self.down4 = down(inner_channels*8, inner_channels*8)
        self.up1 = up(inner_channels*16, inner_channels*4)
        self.up2 = up(inner_channels*8, inner_channels*2)
        self.up3 = up(inner_channels*4, inner_channels)
        self.up4 = up(inner_channels*2, inner_channels)
        self.outc = outconv(inner_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class UNet1d(nn.Module):
    def __init__(self, n_channels, n_classes, inner_channels=64):
        super(UNet1d, self).__init__()
        self.inc = inconv(n_channels, inner_channels, dim=1)
        self.down1 = down(inner_channels, inner_channels*2, dim=1)
        self.down2 = down(inner_channels*2, inner_channels*4, dim=1)
        self.down3 = down(inner_channels*4, inner_channels*8, dim=1)
        self.down4 = down(inner_channels*8, inner_channels*8, dim=1)
        self.up1 = up(inner_channels*16, inner_channels*4, dim=1)
        self.up2 = up(inner_channels*8, inner_channels*2, dim=1)
        self.up3 = up(inner_channels*4, inner_channels, dim=1)
        self.up4 = up(inner_channels*2, inner_channels, dim=1)
        self.outc = outconv(inner_channels, n_classes, dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

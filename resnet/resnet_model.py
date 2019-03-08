# full assembly of the sub-parts to form the complete net
from .resnet_parts import *


class ResNet2d(nn.Module):
    def __init__(self, ic=120, oc=25, s_filter=5):
        super(ResNet2d, self).__init__()
        blocks1 = [nn.Conv2d(ic, 40, s_filter, padding=s_filter // 2)] + 5*[residual_block(40, s_filter, dim=2)]
        blocks2 = [nn.Conv2d(40, 50, s_filter, padding=s_filter // 2)] + 5*[residual_block(50, s_filter, dim=2)]
        blocks3 = [nn.Conv2d(50, 60, s_filter, padding=s_filter // 2)] + 5*[residual_block(60, s_filter, dim=2)]
        blocks4 = [nn.Conv2d(60, 70, s_filter, padding=s_filter // 2)] + 5*[residual_block(70, s_filter, dim=2)]
        blocks5 = [nn.Conv2d(70, 80, s_filter, padding=s_filter // 2)] + 5*[residual_block(80, s_filter, dim=2)]
        blocks = blocks1 + blocks2 + blocks3 + blocks4 + blocks5
        self.blocks = nn.Sequential(*blocks)
        self.out_conv = outconv(80, oc)

    def forward(self, x):
        x = self.blocks(x)
        x = self.out_conv(x)
        x = x.add_(x.transpose(2, 3)).div_(2)
        return x


class ResNet1d(nn.Module):
    def __init__(self, oc=40, n_blocks=6):
        super(ResNet1d, self).__init__()
        blocks = [residual_block(40, s_filter=17, dim=1)] * n_blocks
        self.blocks = nn.Sequential(*blocks)
        self.out_conv = outconv(40, oc, dim=1)

    def forward(self, x):
        x = self.blocks(x)
        x = self.out_conv(x)
        return x

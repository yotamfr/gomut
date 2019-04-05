# full assembly of the sub-parts to form the complete net
from .resnet_parts import *


class ResNet2d(nn.Module):
    def __init__(self, ic=120, hc=40, s_filter=5, num_num_blocks=5):
        super(ResNet2d, self).__init__()
        i = 0
        blocks = []
        while i < num_num_blocks:
            blocks += [nn.Conv2d(ic, hc, s_filter, padding=s_filter // 2)]
            blocks += 5*[residual_block(hc, s_filter, dim=2)]
            ic = hc
            hc += 5
            i += 1
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNet1d(nn.Module):
    def __init__(self, input_size=40, output_size=40, n_blocks=3):
        super(ResNet1d, self).__init__()
        blocks = [residual_block(input_size, s_filter=17, dim=1)] * n_blocks
        self.blocks = nn.Sequential(*blocks)
        self.out_conv = outconv(input_size, output_size, dim=1)

    def forward(self, x):
        x = self.blocks(x)
        x = self.out_conv(x)
        return x

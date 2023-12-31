# -*- coding:utf-8 -*-

"""
# Input 3,256,256
# Encoder
    # Stem
        # Conv 64,128,128
        # BN
        # ReLu
        # MaxPool 64,64,64
    # Body
        # ResBlock1 256,64,64
        # ResBlock2 512,32,32
        # ResBlock3 1024,16,16
        # ResBlock4 2048,8,8
# Decoder
    # Body
        # TransposeResBlock1 1024,16,16
        # TransposeResBlock2 512,32,32
        # TransposeResBlock3 256,64,64
    # Head
        # TransposeConv1 64,128,128
        # TransposeConv2 3,256,256
        # BN
# Output 3,256,256
"""


import torch
from torch import nn, Tensor

from .encoder import ResidualEncoder
from .decoder import TransposeDecoder


def init_weighs(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0, std=0.3)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0, std=0.5)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 0.5)


class SRNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = ResidualEncoder()
        self.decoder = TransposeDecoder()
        self.apply(init_weighs)

    def forward(self, x: Tensor):
        r, f1, f2, f3, out = self.encoder(x)
        out = self.decoder(out, f1, f2, f3, r)
        return out


if __name__ == '__main__':
    srnet = SRNet()
    x = torch.rand(size=(1, 3, 256, 256))
    print("in: ", x.shape)
    out = srnet(x)
    print("out: ", out.shape)

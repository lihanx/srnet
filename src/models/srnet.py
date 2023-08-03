# -*- coding:utf-8 -*-

import torch
from torch import nn, Tensor

from encoder import ResidualEncoder
from decoder import TransposeDecoder


class SRNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = ResidualEncoder()
        self.decoder = TransposeDecoder()

    def forward(self, x: Tensor):
        feature_map = self.encoder(x)
        out = self.decoder(feature_map)
        return out


if __name__ == '__main__':
    srnet = SRNet()
    x = torch.rand(size=(1, 3, 256, 256))
    print("in: ", x.shape)
    out = srnet(x)
    print("out: ", out.shape)

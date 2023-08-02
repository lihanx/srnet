# -*- coding:utf-8 -*-

from torch import nn

from .encoder import ResidualEncoder
from .decoder import TransposeConvDecoder


class SRNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = ResidualEncoder()
        self.decoder = ResidualEncoder()
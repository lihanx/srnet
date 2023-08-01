# -*- coding:utf-8 -*-

from torch import nn


class TransposeDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.decoder = None

    def _init_module(self):

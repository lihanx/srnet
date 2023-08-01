# -*- coding:utf-8 -*-

from torch import nn
from torchvision.models.resnet import resnet18


class ResidualEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = self._init_encoder()

    def _init_module(self):
        origin_resnet18 = resnet18(pretrained=False)
        return nn.Sequential(
            *list(origin_resnet18.children())[:-2]
        )

    def forward(self, x):
        out = self.encoder(x)
        return out
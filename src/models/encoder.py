# -*- coding:utf-8 -*-

from torch import nn
from torchvision.models.resnet import resnet18


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

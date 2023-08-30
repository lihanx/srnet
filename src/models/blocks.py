# -*- coding:utf-8 -*-

from typing import Optional, Callable

from torch import Tensor, nn
from torchvision.models.resnet import conv1x1


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class TransposeBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(planes, planes//self.expansion,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(planes, planes//self.expansion, stride)
        self.bn2 = nn.BatchNorm2d(planes//self.expansion)
        self.upsample = upsample
        self.stride = stride
        self.conv3 = conv1x1(int(planes//self.expansion), planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        return out

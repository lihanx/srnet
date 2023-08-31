# -*- coding:utf-8 -*-

from typing import Type, Union
from collections import OrderedDict

from torch import nn, Tensor
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1


class ResidualEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self._norm_layer = nn.BatchNorm2d
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            self._norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.res_1 = self._make_layer(Bottleneck, 64, 2)
        self.res_2 = self._make_layer(Bottleneck, 128, 2, stride=2)
        self.res_3 = self._make_layer(Bottleneck, 256, 2, stride=2)
        self.res_4 = self._make_layer(Bottleneck, 512, 2, stride=2)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)


    def forward(self, x: Tensor):
        out = self.stem(x)
        out = f1 = self.res_1(out)
        out = f2 = self.res_2(out)
        out = f3 = self.res_3(out)
        out = self.res_4(out)
        return f1, f2, f3, out


if __name__ == '__main__':
    encoder = ResidualEncoder()
    print(encoder)
    import torch
    x = torch.rand(size=(1, 3, 256, 256))
    print(encoder.stem(x).shape)
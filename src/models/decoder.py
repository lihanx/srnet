# -*- coding:utf-8 -*-

from typing import Type
from collections import OrderedDict

import torch
from torch import nn, Tensor

from blocks import TransposeBottleneck


class TransposeDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.inplanes = 2048
        self.base_width = 2048
        self.groups = 1
        self.contraction = 1
        self._norm_layer = nn.BatchNorm2d
        self.body = nn.Sequential(OrderedDict([
            ("TransposeBottleneck1", self._make_transpose(TransposeBottleneck, 2048, 2, stride=2)),
            ("TransposeBottleneck2", self._make_transpose(TransposeBottleneck, 1024, 2, stride=2)),
            ("TransposeBottleneck3", self._make_transpose(TransposeBottleneck, 512, 2, stride=2)),
        ]))
        self.head = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, bias=True),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, bias=True),
            self._norm_layer(3)
        )

    def _make_transpose(
        self,
        block: Type[TransposeBottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        if stride != 1 or self.inplanes != int(planes * block.contraction):
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, int(planes * block.contraction), kernel_size=4, stride=2, padding=1),
                norm_layer(int(planes * block.contraction)),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, upsample, self.groups, self.base_width, norm_layer
            )
        )
        self.inplanes = int(planes * block.contraction)
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        out = self.body(x)
        out = self.head(out)
        return out

if __name__ == '__main__':
    decoder = TransposeDecoder()
    print(decoder)
    x = torch.rand(size=(1, 2048, 8, 8))
    print(decoder(x).shape)
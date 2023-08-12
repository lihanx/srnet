# -*- coding:utf-8 -*-

from typing import Type
from collections import OrderedDict

import torch
from torch import nn, Tensor

from .blocks import TransposeBasicBlock


class TransposeDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.inplanes = 512
        self.base_width = 512
        self.groups = 1
        self.contraction = 1
        self._norm_layer = nn.BatchNorm2d
        self.up_res1 = self._make_transpose(TransposeBasicBlock, 256, 2, stride=2)
        self.up_res2 = self._make_transpose(TransposeBasicBlock, 128, 2, stride=2)
        # self.up_res3 = self._make_transpose(TransposeBasicBlock, 256, 2, stride=2)
        self.up_res3 = self._make_transpose(TransposeBasicBlock, 64, 2, stride=2)

        self.concat_1 = self. _make_combine(256*2, 256)
        self.concat_2 = self. _make_combine(128*2, 128)
        # self.concat_3 = self. _make_combine(256*2, 256)
        self.concat_3 = self. _make_combine(64*2, 64)

        self.head = nn.Sequential(
            nn.ConvTranspose2d(self.inplanes, 64, kernel_size=2, stride=2, bias=True),
            self._norm_layer(64),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, bias=True),
            self._norm_layer(3)
        )

    def _make_combine(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def _make_transpose(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, f1: Tensor, f2: Tensor, f3: Tensor):
        out = self.up_res1(x)
        # print("up res1:", out.shape)
        out = self.concat_1(torch.concat([out, f3], dim=1))

        out = self.up_res2(out)
        # print("up res2:", out.shape)
        out = self.concat_2(torch.concat([out, f2], dim=1))

        out = self.up_res3(out)
        # print("up res3:", out.shape)
        out = self.concat_3(torch.concat([out, f1], dim=1))

        # out = self.up_res4(out)

        out = self.head(out)
        return out

if __name__ == '__main__':
    decoder = TransposeDecoder()
    x = torch.rand(size=(1, 2048, 8, 8))
    print(decoder(x).shape)
# -*- coding:utf-8 -*-

from typing import Type
from collections import OrderedDict

import torch
from torch import nn, Tensor

from .blocks import TransposeBasicBlock


class TransposeDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.inplanes = 2048
        self.base_width = 2048
        self.groups = 1
        self.contraction = 1
        self._norm_layer = nn.BatchNorm2d
        self._relu = nn.PReLU()
        self.up_res1 = self._make_transpose(TransposeBasicBlock, 1024, 2, stride=2)
        self.up_res2 = self._make_transpose(TransposeBasicBlock, 512, 2, stride=2)
        self.up_res3 = self._make_transpose(TransposeBasicBlock, 256, 2, stride=2)

        self.concat_1 = self. _make_combine(1024*2, 1024)
        self.concat_2 = self. _make_combine(512*2, 512)
        self.concat_3 = self. _make_combine(256*2, 256)
        self.concat_4 = self. _make_combine(64*2, 64)

        self.head = nn.Sequential(
            self._make_transpose(TransposeBasicBlock, 128, 2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False),
            self._relu,
        )
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, bias=False),
            nn.Sigmoid(),
        )

    def _make_combine(self, inplanes, planes):
        return nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=True),
            self._relu,
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

    def forward(self, x: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, r: Tensor):
        out = self.up_res1(x)
        out = self.concat_1(torch.concat([out, f3], dim=1))

        out = self.up_res2(out)
        out = self.concat_2(torch.concat([out, f2], dim=1))

        out = self.up_res3(out)
        out = self.concat_3(torch.concat([out, f1], dim=1))

        out = self.head(out)
        out = self.concat_4(torch.concat([out, r], dim=1))

        out = self.final_conv(out)
        return out

if __name__ == '__main__':
    decoder = TransposeDecoder()
    x = torch.rand(size=(1, 2048, 8, 8))
    print(decoder(x).shape)
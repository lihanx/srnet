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
        self.up_res1 = self._make_transpose(TransposeBasicBlock, 1024, 2, stride=2)
        self.up_res2 = self._make_transpose(TransposeBasicBlock, 512, 2, stride=2)
        self.up_res3 = self._make_transpose(TransposeBasicBlock, 256, 2, stride=2)
        self.up_res4 = self._make_transpose(TransposeBasicBlock, 64, 2, stride=1)

        self.concat_1 = self. _make_combine(1024*2, 1024)
        self.concat_2 = self. _make_combine(512*2, 512)
        self.concat_3 = self. _make_combine(256*2, 256)
        self.concat_4 = self. _make_combine(64*2, 64)

        self.head = nn.Sequential(
            nn.ConvTranspose2d(self.inplanes, 64, kernel_size=2, stride=2, bias=False),
            self._norm_layer(64),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, bias=True),
            self._norm_layer(3)
        )

    def _make_combine(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            # nn.ReLU(inplace=True)
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

    def forward(self, x: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor):
        out = self.up_res1(x)
        print("Decoder:", out.shape)
        out = self.concat_1(torch.concat([out, f4], dim=1))

        out = self.up_res2(out)
        print("Decoder:", out.shape)
        out = self.concat_2(torch.concat([out, f3], dim=1))

        out = self.up_res3(out)
        print("Decoder:", out.shape)
        out = self.concat_3(torch.concat([out, f2], dim=1))

        out = self.up_res4(out)
        print("Decoder:", out.shape)
        out = self.concat_4(torch.concat([out, f1], dim=1))
        out = self.head(out)
        return out

if __name__ == '__main__':
    decoder = TransposeDecoder()
    x = torch.rand(size=(1, 2048, 8, 8))
    f3 = torch.rand(size=(1, 1024, 16, 16))
    f2 = torch.rand(size=(1, 512, 32, 32))
    f1 = torch.rand(size=(1, 64, 64, 64))
    print(decoder(x, f1, f2, f3).shape)
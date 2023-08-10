# -*- coding:utf-8 -*-

import os

from netplot.blocks import block_Res, block_Unconv
from netplot.tikzeng import (to_head, to_cor, to_begin, to_end,
                             to_input, to_ConvConvRelu, to_Pool, to_Sum, to_Conv, to_connection, to_skip,
                             to_generate)


arch = [
    # tex header
    to_head("."),
    to_cor(),
    to_begin(),

    # input
    to_input("./noao-n7000mosblock-origin.png", to="(-2,0,0)"),  # 3,256,256

    # encoder
    # stem
    to_ConvConvRelu(name="Stem", s_filer=256, n_filer=(64, 64), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40),  # 64,128,128
    to_Pool("StemPool", offset="(0,0,0)", to="(1,0,0)", width=1, height=20, depth=20, opacity=0.5),  # 64,64,64

    # body
    to_Conv(name="BodyResBlock_1", s_filer=256, n_filer=64, offset="(1,0,0)", to="(2,0,0)", width=2, height=20, depth=20, caption="ResBlock1"),  # 256,64,64
    to_Conv(name="BodyResBlock_2", s_filer=512, n_filer=32, offset="(2,0,0)", to="(3,0,0)", width=4, height=10, depth=10, caption="ResBlock2"),  # 512,32,32
    to_Conv(name="BodyResBlock_3", s_filer=1024, n_filer=16, offset="(3,0,0)", to="(4,0,0)", width=8, height=5, depth=5, caption="ResBlock3"),  # 1024,16,16
    to_Conv(name="BodyResBlock_4", s_filer=2048, n_filer=8, offset="(4,0,0)", to="(5.2,0,0)", width=16, height=2.5, depth=2.5, caption="ResBlock4"),  # 2048,8,8

    # decoder
    # body
    to_Conv(name="UpResBlock_1", s_filer=1024, n_filer=16, offset="(6,0,0)", to="(8,0,0)", width=8, height=5, depth=5, caption="TransposeResBlock1"),  # 1024,16,16
    to_Conv(name="UpResBlock_2", s_filer=512, n_filer=32, offset="(8,0,0)", to="(10,0,0)", width=4, height=10, depth=10, caption="TransposeResBlock2"),  # 512,32,32
    to_Conv(name="UpResBlock_3", s_filer=256, n_filer=64, offset="(10,0,0)", to="(12,0,0)", width=2, height=20, depth=20, caption="TransposeResBlock3"),  # 256,64,64

    # head
    to_Conv("HeadTranspose_1", offset="(12,0,0)", to="(14,0,0)", width=1, height=20, depth=20),  # 64,64,64
    to_ConvConvRelu(name="HeadTranspose_2", s_filer=256, n_filer=(64, 64), offset="(14,0,0)", to="(16,0,0)", width=(2,2), height=40, depth=40),  # 3,256,256

    # output
    to_input("./noao-n7000mosblock-reduced.png", to="(33,0,0)"),  # 3,256,256

    # arrows
    to_connection(of="StemPool", to="BodyResBlock_1"),
    to_connection(of="BodyResBlock_1", to="BodyResBlock_2"),
    to_connection(of="BodyResBlock_2", to="BodyResBlock_3"),
    to_connection(of="BodyResBlock_3", to="BodyResBlock_4"),
    to_connection(of="BodyResBlock_4", to="UpResBlock_1"),
    to_connection(of="UpResBlock_1", to="UpResBlock_2"),
    to_connection(of="UpResBlock_2", to="UpResBlock_3"),
    to_connection(of="UpResBlock_3", to="HeadTranspose_1"),
    to_connection(of="HeadTranspose_1", to="HeadTranspose_2"),

    # sum
    to_Sum(name="sum1", offset="(8.4,0,0)", to="(8.4,0,0)", radius=1.5, opacity=0.6),
    to_Sum(name="sum2", offset="(10.2,0,0)", to="(10.2,0,0)", radius=1.5, opacity=0.6),
    to_Sum(name="sum3", offset="(12.1,0,0)", to="(12.1,0,0)", radius=1.5, opacity=0.6),

    # skip
    to_skip(of="BodyResBlock_3", to="sum1", pos=3),
    to_skip(of="BodyResBlock_2", to="sum2", pos=4),
    to_skip(of="BodyResBlock_1", to="sum3", pos=5),

    # tex end
    to_end(),
]


if __name__ == '__main__':
    idx = __file__.rfind(".")
    filename = __file__[:idx]
    print(f"Generating {filename}.tex...")
    to_generate(arch, pathname=f"{filename}.tex")
    print("Done.")
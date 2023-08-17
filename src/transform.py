# -*- coding:utf-8 -*-
import logging
import sys
import os
from argparse import ArgumentParser

from PIL import Image
import torch
from torch import Tensor

from torchvision.transforms import functional as F

from .models.srnet import SRNet


class SRNetTransformer:

    def __init__(self, weight_name: str, log_level):
        self.inplanes = 256
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = SRNet()
        cwd = os.getcwd()
        weight_path = os.path.join(cwd, f"weights/{weight_name}")
        self.net.load_state_dict(torch.load(weight_path))
        self.net.eval()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

    def pad_image(self, img: Tensor) -> Tensor:
        w, h = img.shape
        if w >= self.inplanes:
            wp = w % self.inplanes
        else:
            wp = self.inplanes - w
        if h >= self.inplanes:
            hp = h % self.inplanes
        else:
            hp = self.inplanes - h
        img = F.pad(img, [0, 0, wp, hp])  # left, top, right, bottom
        return img

    def transform(self, image_path, output_path):
        with Image.open(image_path) as img:
            img = img.convert(mode="RGB")
            img_tensor = F.to_tensor(img)
            padded = self.pad_image(img_tensor)
            # crop
            # normalize
            # inference
            # concat
            # unpadding
            # save



def parse_inference_args(args):
    parser = ArgumentParser()
    parser.add_argument("--image", action="store_const", required=True)
    parser.add_argument("--output", action="store_const", required=False, default="")
    parser.add_argument("--weight", action="store_const", required=False, default="v1.pth")
    parser.add_argument("--log_level", action="store_const", default="info")
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_inference_args(sys.argv[1:])
    output = args.output
    if not output:
        image_path, ext = os.path.splitext(args.image)
        output = f"{image_path}_reduced{ext}"
    log_level = args.log_level.upper()
    log_level = getattr(logging, log_level, logging.INFO)
    transformer = SRNetTransformer(args.weight, log_level)
    transformer.transform(image_path=args.image, output_path=output)
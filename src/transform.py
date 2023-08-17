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
        self.net = self.net.to(self.device)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

    def pad_image(self, img: Tensor):
        """padding 以保证图片可以被 256,256 裁切完整覆盖"""
        _, _, w, h = img.shape
        pl, pt, pr, pb = 0, 0, 0, 0
        if w >= self.inplanes:
            pr = w % self.inplanes
        else:
            pr = self.inplanes - w
        if h >= self.inplanes:
            pb = h % self.inplanes
        else:
            pb = self.inplanes - h
        if pl or pt or pr or pb:
            img = F.pad(img, [pl, pt, pr, pb])  # left, top, right, bottom
        return img, pl, pt, pr, pb

    def normalize(self, img_tensor: Tensor):
        """normalize inplace"""
        return img_tensor.float().div(255.0)

    def generate_cropped_input(self, img_tensor: Tensor):
        nw, nh = 0, 0
        _, _, w, h = img_tensor.shape
        lx, ly = 0, 0
        while lx < w and ly < h:
            lx, ly = nw * self.inplanes, nh * self.inplanes
            cropped = F.crop(img_tensor, lx, ly, self.inplanes, self.inplanes)
            yield cropped, lx, ly

    def transform(self, image_path, output_path):
        with Image.open(image_path) as img:
            img = img.convert(mode="RGB")
            img_tensor = F.to_tensor(img)
            self.normalize(img_tensor)
            # padding
            padded, pl, pt, pr, pb = self.pad_image(img_tensor)
            # to same device
            padded = padded.to(self.device)
            new_img_tensor = torch.zeros_like(padded).to(self.device)
            # crop
            for cropped, lx, ly in self.generate_cropped_input(img_tensor):
                # inference
                transformed = self.net(cropped)
                # copy 到新 tensor 的对应位置
                # concat
                new_img_tensor[0, 0:3, lx:lx+self.inplanes, ly:ly+self.inplanes] = transformed[0, 0:3, lx:lx+self.inplanes, ly:ly+self.inplanes]
            # remove padding
            new_img_tensor = F.crop(new_img_tensor, 0+pt, 0+pl, img.height, img.width)
            new_img = F.to_pil_image(new_img_tensor, mode="RGB")
            # save
            new_img.save(output_path)
        return None


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
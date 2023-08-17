# -*- coding:utf-8 -*-
import logging
logging.basicConfig(level=logging.INFO)
from typing import Union
import sys
import os
from argparse import ArgumentParser

from PIL import Image
import torch
from torch import Tensor
from torchvision.transforms import functional as F

from models.srnet import SRNet
from utils import _save_weight_from_checkpoint


logger = logging.getLogger(__file__)


class SRNetTransformer:

    def __init__(self,
                 weight_name: Union[str, None],
                 checkpoint_name: Union[str, None] = None):
        self.inplanes = 256
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = SRNet()
        cwd = os.getcwd()
        if weight_name is not None:
            weight_path = os.path.join(cwd, f"weights/{weight_name}")
            self.net.load_state_dict(torch.load(weight_path))
        elif checkpoint_name is not None:
            ckpt_path = os.path.join(cwd, f"checkpoints/{checkpoint_name}")
            ckpt = torch.load(ckpt_path)
            self.net.load_state_dict(ckpt["model_state_dict"])
            weight_name = checkpoint_name.replace(".ckpt", "pth")
            weight_path = os.path.join(cwd, f"weights/{weight_name}")
            _save_weight_from_checkpoint(ckpt, weight_path)
        else:
            raise ValueError("Required at least 1 argument: weight or checkpoint.")
        self.net.eval()
        self.net = self.net.to(self.device)

    def pad_image(self, img: Tensor):
        """padding 以保证图片可以被 256,256 裁切完整覆盖"""
        _, _, w, h = img.shape
        pl, pt, pr, pb = 0, 0, 0, 0
        pad_w = (self.inplanes - w % self.inplanes)  # w padding
        pad_h = (self.inplanes - h % self.inplanes)  # h padding
        if pad_w or pad_h:
            pl = pad_w // 2  # left
            pt = pad_h // 2  # top
            pr = pad_w - pl  # right
            pb = pad_h - pt  # bottom
            img = F.pad(img, [pl, pt, pr, pb])  # left, top, right, bottom
        return img, pl, pt, pr, pb

    def normalize(self, img_tensor: Tensor):
        """normalize inplace"""
        return img_tensor.float().div(255.0)

    def generate_cropped_input(self, img_tensor: Tensor):
        _, _, w, h = img_tensor.shape
        for x in range(w//self.inplanes):
            for y in range(h//self.inplanes):
                pos_x, pos_y = x * self.inplanes, y * self.inplanes
                cropped = F.crop(img_tensor, pos_x, pos_y, self.inplanes, self.inplanes)
                yield cropped, pos_x, pos_y

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
            new_img_tensor = F.crop(new_img_tensor, pt, pl, img.height, img.width)
            new_img = F.to_pil_image(new_img_tensor, mode="RGB")
            # save
            new_img.save(output_path)
        return None


def parse_inference_args(args):
    parser = ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=False, default="")
    parser.add_argument("--weight", required=False, default=None)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_inference_args(sys.argv[1:])
    output = args.output
    if not output:
        image_path, ext = os.path.splitext(args.image)
        output = f"{image_path}_reduced{ext}"
    transformer = SRNetTransformer(args.weight, args.checkpoint)
    transformer.transform(image_path=args.image, output_path=output)
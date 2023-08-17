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
                 checkpoint_name: Union[str, None] = None,
                 device_name: Union[str, None] = "cuda"):
        self.inplanes = 256
        self.device = torch.device(device_name)
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
        _, _, h, w = img.shape
        logger.info(f"Image Size: {w},{h}")
        pl, pt, pr, pb = 0, 0, 0, 0
        pad_w = 0 if w % self.inplanes == 0 else self.inplanes - (w % self.inplanes) # w padding
        pad_h = 0 if h % self.inplanes == 0 else self.inplanes - (h % self.inplanes) # h padding
        if pad_w:
            pl = pad_w // 2  # left
            pr = pad_w - pl  # right
        if pad_h:
            pt = pad_h // 2  # top
            pb = pad_h - pt  # bottom
        if pl or pt or pr or pb:
            img = F.pad(img, [pl, pt, pr, pb])  # left, top, right, bottom
        return img, pl, pt, pr, pb

    def generate_cropped_input(self, img_tensor: Tensor):
        _, _, w, h = img_tensor.shape
        for y in range(h//self.inplanes):
            batch = []
            pos_y = y * self.inplanes
            for x in range(w//self.inplanes):
                pos_x = x * self.inplanes
                cropped = F.crop(img_tensor, pos_x, pos_y, self.inplanes, self.inplanes)
                batch.append(cropped)
            yield torch.concat(batch, dim=0), pos_y

    def transform(self, image_path, output_path):
        with Image.open(image_path) as img:
            logger.info(f"Open image: {image_path}.")
            img = img.convert(mode="RGB")
            img_tensor = F.to_tensor(img).unsqueeze(0)
            # padding
            padded, pl, pt, pr, pb = self.pad_image(img_tensor)
            _, c, h, w = padded.shape
            # to same device
            new_img_tensor = torch.zeros(size=(3, h, w))
            # crop
            for batch, pos_y in self.generate_cropped_input(padded):
                # inference
                batch = batch.to(self.device)
                transformed = self.net(batch)
                logger.info(f"Row {pos_y} transformed.")
                # copy 到新 tensor 的对应位置
                # concat
                for idx, p in enumerate(transformed):
                    pos_x = idx * self.inplanes
                    new_img_tensor[:, pos_x:pos_x+self.inplanes, pos_y:pos_y+self.inplanes] = p[:, :, :]
            # remove padding
            new_img_tensor = F.crop(new_img_tensor, pt, pl, img.height, img.width)
            new_img = F.to_pil_image(new_img_tensor, mode="RGB")
            # save
            new_img.save(output_path)
            logger.info(f"Save image: {output_path}.")
            logger.info("Done.")
        return None


def parse_inference_args(args):
    parser = ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=False, default="")
    parser.add_argument("--weight", required=False, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_inference_args(sys.argv[1:])
    output = args.output
    if not output:
        image_path, ext = os.path.splitext(args.image)
        output = f"{image_path}_reduced{ext}"
    transformer = SRNetTransformer(args.weight, args.checkpoint, args.device)
    transformer.transform(image_path=args.image, output_path=output)
# -*- coding:utf-8 -*-

import os
import logging

import torch
from torchvision.transforms import functional as F

from loss.ssim import SSIMFunc


logger = logging.getLogger(__name__)


def _check_dir(dirpath):
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
        logger.info(f"Create directory {dirpath}.")


def _save_weight_from_checkpoint(ckpt, weight_path):
    torch.save(ckpt["model_state_dict"], weight_path)
    logger.info(f"Model weights saved: {weight_path}")


def get_psnr(img1: torch.Tensor, img2: torch.Tensor):
    """
    PSNR接近 50dB ，代表压缩后的图像仅有些许非常小的误差。
    PSNR大于 30dB ，人眼很难察觉压缩后和原始影像的差异。
    PSNR介于 20dB 到 30dB 之间，人眼就可以察觉出图像的差异。
    PSNR介于 10dB 到 20dB 之间，人眼还是可以用肉眼看出这个图像原始的结构，且直观上会判断两张图像不存在很大的差异。
    PSNR低于 10dB，人类很难用肉眼去判断两个图像是否为相同，一个图像是否为另一个图像的压缩结果。
    Tensor value in [0.0, 1.0]
    """
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(torch.max(img1).item() * 2 / mse)


def save_puzzles(img: torch.Tensor, output_path, idx):
    filepath, filename = os.path.split(output_path)
    name, ext = os.path.splitext(filename)
    dirpath = os.path.join(filepath, f"{name}_puzzles")
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    output_name = f"{name}_{idx}{ext}"
    output_path = os.path.join(dirpath, output_name)
    pil_img = F.to_pil_image(img, mode="RGB")
    pil_img.save(output_path)
    logger.info(f"{output_path} saved.")


get_ssim = SSIMFunc(window_size=11)
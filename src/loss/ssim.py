# -*- coding:utf-8 -*-
"""
# ============================================
# SSIM loss
# https://github.com/Po-Hsun-Su/pytorch-ssim
# ============================================

使用新的 Tensor.requires_grad_() API 实现
"""

from math import exp

import torch
from torch import Tensor
import torch.nn.functional as F


class Window:

    @staticmethod
    def gaussian(window_size: int, sigma: float):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    @classmethod
    def create(cls, window_size: int, channel: int, sigma: float = 1.5):
        _1D_window = cls.gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size)\
            .contiguous()\
            .requires_grad_()
        return window


class SSIMFunc:

    def __init__(self, window_size: int, size_average: bool = True):
        self.window_size = window_size
        self.size_average = size_average
        self._window = None

    def __call__(self, image1: Tensor, image2: Tensor):
        if len(image1.shape) != 4:
            raise ValueError(f"Invalid Input Image({image1.shape}), 4 dimensions required.")
        _, channel, _, _ = image1.shape

        if self._window is None:
            self._window = Window.create(self.window_size, channel)
        self._window = self._window.to(image1.device)
        mu1 = F.conv2d(image1, self._window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(image2, self._window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(image1 * image1, self._window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(image2 * image2, self._window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(image1 * image2, self._window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.ssim_fn = SSIMFunc(window_size, size_average)

    def forward(self, image: Tensor, target: Tensor):
        return 1 - self.ssim_fn(image, target)


if __name__ == '__main__':
    from PIL import Image
    from torchvision.transforms import ToPILImage, ToTensor
    from torch.optim import Adam
    import torch

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    to_tensor = ToTensor()
    to_pil = ToPILImage()
    ssim_fn = SSIMFunc(window_size=11)

    with Image.open("./einstein.png") as image_pil:
        target = to_tensor(image_pil).unsqueeze(0)

        image = torch.rand(size=target.shape).requires_grad_()
        print(image)

        ssim_value = ssim_fn(image, target)
        print(f"SSIM Value: {ssim_value}")

        ssim_loss_fn = SSIMLoss()
        optimizer = Adam([image], lr=0.01)
        last_ssim = 0
        while abs(ssim_value) < 0.98:
            optimizer.zero_grad()
            ssim_loss = ssim_loss_fn(image, target)
            ssim_value = ssim_fn(image, target)
            print(f"SSIM Value: {ssim_value}")
            ssim_loss.backward()
            optimizer.step()
            if abs(ssim_value - last_ssim) > 0.02:
                img_pil = to_pil(image.squeeze(0))
                img_pil.show()
                last_ssim = ssim_value


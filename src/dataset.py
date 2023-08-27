# -*- coding:utf-8 -*-

import os
import logging
import random
import time
from typing import Tuple, Sequence, List, Union

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.transforms import RandomCrop, RandomRotation, ColorJitter, RandomResizedCrop


logger = logging.getLogger(__name__)


class RandAugmentationDataSet(Dataset):

    def __init__(self,
                 path: str,
                 origin_dir: str,
                 reduced_dir: str,
                 limit: int,
                 output_size: Tuple[int, int] = (256, 256),
                 hflip_p: float = 0.5,
                 vflip_p: float = 0.5,
                 rotation_p: float = 0.4,
                 rotation_degrees: Union[List[int], None] = None,
                 color_distortion_p: float = 0.7):
        # dataset params
        self.path = os.path.abspath(path)
        self.origin_path = os.path.join(self.path, origin_dir)
        self.reduced_path = os.path.join(self.path, reduced_dir)
        self._image_list = None
        self.limit = limit
        self.output_size = output_size
        self._current = 1
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # augmentation prob
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rotation_p = rotation_p
        self.color_distortion_p = color_distortion_p
        # rotation params
        self.rotate_degrees = [30, 60, 90, 120, 150, 180, 210, 240, 270] if rotation_degrees is None else rotation_degrees
        self.rotate_options = {
            "interpolation": F.InterpolationMode.BILINEAR,
            "expand": False,
            "center": None,
            "fill": 0,
        }
        # color params
        self.color_options = {
            "brightness": None,
            "contrast": None,
            "saturation": None,
            "hue": [-0.5, 0.5]
        }
        # noise params
        self.noise_options = {
            "mean": 0,
            "std": 0.03,
        }
        # resize params
        self.resizecrop_options = {
            "ratio": [3.0 / 4.0, 4.0 / 3.0],
            "scale": [0.8, 1.2],
        }

    @property
    def image_list(self):
        if self._image_list is None:
            self._image_list = [name for name in os.listdir(self.origin_path)
                      if name.lower().endswith(("tiff", "tif", "png"))
                      and os.path.isfile(f"{self.origin_path}{os.path.sep}{name}")]
        if not self._image_list:
            raise ValueError(f"Empty DataSet {self.origin_path}")
        return self._image_list

    def _rand_crop(self, origin: Tensor, reduced: Tensor):
        """随机裁切"""
        options = RandomCrop.get_params(origin, self.output_size)
        logger.debug(f"Crop: {options}")
        return F.crop(origin, *options), F.crop(reduced, *options)

    def _rand_hflip(self, origin: Tensor, reduced: Tensor):
        """随机水平翻转"""
        if torch.rand(1) < self.hflip_p:
            logger.debug(f"HFlip")
            return F.hflip(origin), F.hflip(reduced)
        return origin, reduced

    def _rand_vflip(self, origin: Tensor, reduced: Tensor):
        """随机垂直翻转"""
        if torch.rand(1) < self.vflip_p:
            logger.debug(f"VFlip")
            return F.vflip(origin), F.vflip(reduced)
        return origin, reduced

    def _rand_rotate(self, origin: Tensor, reduced: Tensor):
        """随机旋转"""
        if torch.rand(1) > self.rotation_p:
            return origin, reduced
        angle = RandomRotation.get_params(self.rotate_degrees)
        channels, _, _ = F.get_dimensions(origin)
        fill = self.rotate_options["fill"]
        if isinstance(origin, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif isinstance(fill, Sequence):
                fill = [float(f) for f in fill]
            else:
                raise TypeError("Fill should be either a sequence or a number.")
        options = self.rotate_options.copy()
        options.update({
            "angle": angle,
            "fill": fill,
        })
        logger.debug(f"Rotate: {angle}")
        return F.rotate(origin, **options), F.rotate(reduced, **options)

    def _rand_noise(self, origin: Tensor, reduced: Tensor):
        # 高斯噪声
        if torch.rand(1) < 0.5:
            g_origin: Image = F.to_pil_image(origin, mode="RGB")
            g_origin = g_origin.convert("L")
            g_tensor = F.to_tensor(g_origin)
            max_val = torch.max(g_tensor)
            mask = (g_tensor < 0.5 * max_val).int() * (g_tensor > 0.2 * max_val).int()
            _, h, w = origin.shape
            _noise = torch.normal(self.noise_options["mean"], self.noise_options["std"], size=(1, h, w))
            _noise *= mask
            noise = torch.zeros_like(origin)
            noise[random.randrange(3),:,:] = _noise
            noise = noise.to(origin.device)
            origin.add_(noise)
        return origin, reduced

    def _rand_color_distort(self, origin: Tensor, reduced: Tensor):
        """随机色差"""
        if torch.rand(1) < 0.7:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = ColorJitter.get_params(**self.color_options)
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    origin = F.adjust_brightness(origin, brightness_factor)
                    reduced = F.adjust_brightness(reduced, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    origin = F.adjust_contrast(origin, contrast_factor)
                    reduced = F.adjust_contrast(reduced, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    origin = F.adjust_saturation(origin, saturation_factor)
                    reduced = F.adjust_saturation(reduced, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    origin = F.adjust_hue(origin, hue_factor)
                    reduced = F.adjust_hue(reduced, hue_factor)
        return origin, reduced

    def _rand_image_file(self):
        """随机获取文件
        train 与 test 同名对应"""
        image_name = random.choice(self.image_list)
        origin_file = os.path.join(self.origin_path, image_name)
        reduced_file = os.path.join(self.reduced_path, image_name)
        if not os.path.exists(reduced_file):
            raise ValueError(f"File {image_name} not exists in {self.reduced_path}")
        return origin_file, reduced_file

    def _rand_resizecrop(self, origin: Tensor, reduced: Tensor):
        i, j, h, w = RandomCrop.get_params(origin, self.output_size)
        if torch.rand(1) < 0.5:
            ratio = torch.empty(1).uniform_(0.6, 1.4).item()
            h = int(h * ratio)
            w = int(w / ratio)
            origin = F.crop(origin, i, j, h, w)
            origin = F.resize(origin, list(self.output_size), antialias=None)
            reduced = F.crop(reduced, i, j, h, w)
            reduced = F.resize(reduced, list(self.output_size), antialias=None)
        else:
            origin, reduced = F.crop(origin, i, j, h, w), F.crop(reduced, i, j, h, w)
        return origin, reduced

    def to_tensor(self, origin: Image, reduced: Image):
        """PIL Image 转换为 Tensor"""
        return F.to_tensor(origin), F.to_tensor(reduced)

    def rand_transform(self, origin: Tensor, reduced: Tensor):
        # 缩放裁切 (256, 256)
        origin, reduced = self._rand_resizecrop(origin, reduced)
        # 高斯噪声
        origin, reduced = self._rand_noise(origin, reduced)
        # 随机水平翻转
        origin, reduced = self._rand_hflip(origin, reduced)
        # 随机垂直翻转
        origin, reduced = self._rand_vflip(origin, reduced)
        # 随机旋转特定角度
        origin, reduced = self._rand_rotate(origin, reduced)
        # 随机转色
        origin, reduced = self._rand_color_distort(origin, reduced)
        return origin, reduced

    def __len__(self):
        return self.limit

    def __getitem__(self, item):
        origin_file, reduced_file = self._rand_image_file()
        with Image.open(origin_file) as origin_pil, \
                Image.open(reduced_file) as reduced_pil:
            origin, reduced = self.to_tensor(origin_pil, reduced_pil)
            origin, reduced = origin.to(self.device), reduced.to(self.device)
            origin, reduced = self.rand_transform(origin, reduced)
            return origin, reduced


if __name__ == '__main__':
    cwd = os.path.abspath(os.path.dirname(__file__))
    image_dir = "images"
    dataset_dir = os.path.join(cwd, image_dir)
    dataset = RandAugmentationDataSet(path=dataset_dir, origin_dir="origin", reduced_dir="reduced", limit=1)
    from torchvision.transforms import functional as F
    # for origin, reduced in dataset:
    for i in range(20):
        origin, reduced = dataset[i]
        print(origin.shape, reduced.shape)
        origin = F.to_pil_image(origin, mode="RGB")
        origin.show()
        reduced = F.to_pil_image(reduced, mode="RGB")
        reduced.show()
        break
    # from torch.utils.data import DataLoader
    #
    # loader = DataLoader(dataset=dataset, batch_size=64)
    # for train, target in loader:
    #     print(len(train), len(target))
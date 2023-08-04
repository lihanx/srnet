# -*- coding:utf-8 -*-

import os
import logging
import random
import time
from typing import Tuple, Sequence, List, Union

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import IterableDataset
from torchvision.transforms import functional as F
from torchvision.transforms import RandomCrop, RandomRotation, ColorJitter


logger = logging.getLogger(__name__)


class RandAugmentationDataSet(IterableDataset):

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
        # augmentation prob
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rotation_p = rotation_p
        self.color_distortion_p = color_distortion_p
        # rotation params
        self.rotate_degrees = [60, 90, 180, 270] if rotation_degrees is None else rotation_degrees
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
            "std": 1,
        }

    @property
    def image_list(self):
        if self._image_list is None:
            self._image_list = [name for name in os.listdir(self.origin_path)
                      if name.lower().endswith(("tiff", "tif"))
                      and os.path.isfile(f"{self.origin_path}{os.path.sep}{name}")]
        if not self._image_list:
            raise ValueError(f"Empty DataSet {self.origin_path}")
        return self._image_list

    def _normalize_input_image(self, origin: Image, reduced: Image):
        """归一化输入
        - 颜色空间
        - 检测尺寸
        - """
        assert origin.size == reduced.size
        return origin, reduced

    def _rand_crop(self, origin: Tensor, reduced: Tensor):
        """随机裁切"""
        options = RandomCrop.get_params(origin, self.output_size)
        logger.info(f"Crop: {options}")
        return F.crop(origin, *options), F.crop(reduced, *options)

    def _rand_hflip(self, origin: Tensor, reduced: Tensor):
        """随机水平翻转"""
        if torch.rand(1) < self.hflip_p:
            logger.info(f"HFlip")
            return F.hflip(origin), F.hflip(reduced)
        return origin, reduced

    def _rand_vflip(self, origin: Tensor, reduced: Tensor):
        """随机垂直翻转"""
        if torch.rand(1) < self.vflip_p:
            logger.info(f"VFlip")
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
        logger.info(f"Rotate: {angle}")
        return F.rotate(origin, **options), F.rotate(reduced, **options)

    def _rand_noise(self, origin: Tensor, reduced: Tensor):
        # 高斯噪声
        noise = torch.normal(self.noise_options["mean"], self.noise_options["std"], size=origin.shape)
        origin = origin.float() + noise
        return origin, reduced.float()

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

    def to_tensor(self, origin: Image, reduced: Image):
        """PIL Image 转换为 Tensor"""
        return F.pil_to_tensor(origin), F.pil_to_tensor(reduced)

    def rand_transform(self, origin: Tensor, reduced: Tensor):
        # 裁切 (256, 256)
        origin, reduced = self._rand_crop(origin, reduced)
        # 随机水平翻转
        origin, reduced = self._rand_hflip(origin, reduced)
        # 随机垂直翻转
        origin, reduced = self._rand_vflip(origin, reduced)
        # 随机旋转特定角度
        # origin, reduced = self._rand_rotate(origin, reduced)
        # 随机转色
        # origin, reduced = self._rand_color_distort(origin, reduced)
        # 高斯噪声
        origin, reduced = self._rand_noise(origin, reduced)
        print(origin, reduced)
        return origin, reduced

    def __iter__(self):
        return self

    def __next__(self):
        if self._current <= self.limit:
            origin_file, reduced_file = self._rand_image_file()
            self._current += 1
            with Image.open(origin_file) as origin_pil, \
                    Image.open(reduced_file) as reduced_pil:
                origin_pil, reduced_pil = self._normalize_input_image(origin_pil, reduced_pil)
                origin, reduced = self.to_tensor(origin_pil, reduced_pil)
                origin, reduced = self.rand_transform(origin, reduced)
                return origin, reduced
        else:
            raise StopIteration


if __name__ == '__main__':
    cwd = os.path.abspath(os.path.dirname(__file__))
    image_dir = "images"
    dataset_dir = os.path.join(cwd, image_dir)
    dataset = RandAugmentationDataSet(path=dataset_dir, origin_dir="origin", reduced_dir="reduced", limit=1)
    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage(mode="RGB")
    for origin, reduced in dataset:
        print(origin.shape, reduced.shape)
        origin = to_pil(origin/255.0)
        origin.show()
        reduced = to_pil(reduced/255.0)
        reduced.show()

    # from torch.utils.data import DataLoader
    #
    # loader = DataLoader(dataset=dataset, batch_size=64)
    # for train, target in loader:
    #     print(len(train), len(target))
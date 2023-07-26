# -*- coding:utf-8 -*-

import os
import logging
import random
from typing import Tuple, Sequence, List, Union

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import IterableDataset
from torchvision.transforms import functional as F
from torchvision.transforms import RandomCrop, RandomRotation


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
        self._current = 0
        # augmentation prob
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rotation_p = rotation_p
        self.color_distortion_p = color_distortion_p
        # rotation params
        self.rotate_degrees = [60, 90, 180, 270] if rotation_degrees is None else rotation_degrees
        self.rotate_options = {
            "interpolation": F.InterpolationMode.BICUBIC,
            "expand": False,
            "center": None,
            "fill": 0,
        }

    @property
    def image_list(self):
        if self._image_list is None:
            self._image_list = [name for name in os.listdir()
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
        assert origin.size() == reduced.size()
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

    def _rand_color_distort(self, origin: Tensor, reduced: Tensor):
        """随机色差"""
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

    def rand_transform(self, origin: Image, reduced: Image):
        # 裁切 (256, 256)
        origin, reduced = self._rand_crop(origin, reduced)
        # 随机水平翻转
        origin, reduced = self._rand_hflip(origin, reduced)
        # 随机垂直翻转
        origin, reduced = self._rand_vflip(origin, reduced)
        # 随机旋转特定角度
        origin, reduced = self._rand_rotate(origin, reduced)
        # 随机转色
        origin, reduced = self._rand_color_distort(origin, reduced)
        return origin, reduced

    def __iter__(self):
        return self

    def __next__(self):
        while self._current < self.limit:
            origin_file, reduced_file = self._rand_image_file()
            with Image.open(origin_file) as origin_pil, \
                    Image.open(reduced_file) as reduced_pil:
                origin_pil, reduced_pil = self._normalize_input_image(origin_pil, reduced_pil)
                origin, reduced = self.to_tensor(origin_pil, reduced_pil)
                origin, reduced = self.rand_transform(origin, reduced)
                yield origin, reduced
            self._current += 1
        else:
            raise StopIteration
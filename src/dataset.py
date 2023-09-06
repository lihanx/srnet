# -*- coding:utf-8 -*-

import os
import logging
import random
import math
from typing import Tuple, Sequence, List, Union

from PIL import Image
import torch
from torch import Tensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, RandomRotation, ColorJitter


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
            "ratio": 0.08,
        }
        # resize params
        self.resizecrop_options = {
            "ratio": [3.0 / 4.0, 4.0 / 3.0],
            "scale": [0.8, 1.2],
        }
        self._origin = []
        self._reduced = []
        self.init_images()
        self._cnt = 0

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
        if torch.rand(1) < 0.6:
            max_val = torch.max(origin)
            median_val = torch.median(origin)
            std = self.noise_options["ratio"] * median_val
            mask = (origin < 0.95 * max_val).int()
            c, h, w = origin.shape
            _noise = torch.normal(self.noise_options["mean"], std, size=(c, h, w)) * math.sqrt(0.5)
            _noise = _noise.to(origin.device)
            mask = mask.to(origin.device)
            _noise *= mask
            origin.add_(_noise)
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

    def _rand_resizecrop(self, origin: Tensor, reduced: Tensor):
        i, j, h, w = RandomCrop.get_params(origin, self.output_size)
        if torch.rand(1) < 0.5:
            ratio = torch.empty(1).uniform_(0.2, 1.5).item()
            h = int(h * ratio)
            w = int(w * ratio)
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

    def init_images(self):
        if self._origin is None:
            self._origin = []
        if self._reduced is None:
            self._reduced = []
        for image_name in self.image_list:
            origin_file = os.path.join(self.origin_path, image_name)
            reduced_file = os.path.join(self.reduced_path, image_name)
            with Image.open(origin_file) as origin, \
                    Image.open(reduced_file) as reduced:
                origin, reduced = self.to_tensor(origin, reduced)
                self._origin.append(origin)
                self._reduced.append(reduced)
        logger.info("Image Tensors init.")

    def _rand_image_tensor(self):
        """随机获取文件
        train 与 test 同名对应"""
        if not self._image_list:
            raise ValueError("Empty Image List")
        idx = random.randrange(len(self.image_list))
        # logger.info(f"Use Image-{idx}")
        return self._origin[idx], self._reduced[idx]

    def __getitem__(self, item):
        origin, reduced = self._rand_image_tensor()
        origin, reduced = self.rand_transform(origin.to(self.device), reduced.to(self.device))
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
        # reduced = F.to_pil_image(reduced, mode="RGB")
        # reduced.show()
        # break
    # from torch.utils.data import DataLoader
    #
    # loader = DataLoader(dataset=dataset, batch_size=64)
    # for train, target in loader:
    #     print(len(train), len(target))
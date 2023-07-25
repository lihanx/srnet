# -*- coding:utf-8 -*-

from torchvision.transforms import (Compose, RandomCrop, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip,
                                    PILToTensor, Normalize, ColorJitter)


rand_transform = Compose([
    RandomCrop,
    RandomRotation,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    PILToTensor,
])
# -*- coding:utf-8 -*-

from torch.utils.data import DataLoader
import torch

from .dataset import RandAugmentationDataSet
from .models import SRNet


class SRNetTrainer:

    def __init__(self):
        self.dataset = RandAugmentationDataSet(path="./images", origin_dir="origin", reduced_dir="reduced", limit=100000)
        self.train_dataloader = DataLoader(self.dataset, batch_size=64)
        self.test_dataloader = DataLoader(self.dataset, batch_size=64)
        self.model = SRNet()

    def train(self):...
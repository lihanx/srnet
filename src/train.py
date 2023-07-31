# -*- coding:utf-8 -*-

from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import functional as F

from .dataset import RandAugmentationDataSet
from .models import SRNet


class SRNetTrainer:

    def __init__(self):
        self.dataset = RandAugmentationDataSet(path="./images", origin_dir="origin", reduced_dir="reduced", limit=100000)
        self.epochs = 100
        self.learning_rate = 0.1
        self.batch_size = 64
        self.train_dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
        self.net = SRNet()
        self.optimizer = SGD(self.net.parameters(), lr=self.learning_rate)

    def loss(self, input, target):
        return F.cross_entropy()

    def train(self):...
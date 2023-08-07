# -*- coding:utf-8 -*-
import os.path

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adagrad, Adam
from torch.optim.lr_scheduler import LRScheduler, MultiplicativeLR, LambdaLR, CosineAnnealingLR
from torch.nn import functional as F

from .dataset import RandAugmentationDataSet
from .models import SRNet
from .loss import SSIMLoss


class SRNetTrainer:

    def __init__(self):
        self.epochs = 1500
        self.learning_rate = 1e-4
        self.batch_size = 64
        cwd = os.path.abspath(os.path.dirname(__file__))
        self.dataset = RandAugmentationDataSet(path=os.path.join(cwd, "images"), origin_dir="origin", reduced_dir="reduced", limit=100000)
        self.train_dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = SRNet()
        self.net.to(self.device)
        self.optimizer = Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_fn = SSIMLoss()
        self.lr_decay_rate = 0.8
        self.lr_epoch_per_decay = 100
        self.scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: self.lr_decay_rate ** (epoch // self.lr_epoch_per_decay))

    def _test(self):...

    def _train(self):...

    def train(self):
        for epoch in range(self.epochs):
            self.scheduler.step(epoch)


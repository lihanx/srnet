# -*- coding:utf-8 -*-
import datetime
import os.path
import logging
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from .dataset import RandAugmentationDataSet
from .models import SRNet
from .loss import SSIMLoss


logger = logging.getLogger(__name__)


class SRNetTrainer:

    def __init__(self):
        self.epochs = 1500
        self.learning_rate = 1e-4
        self.batch_size = 64
        cwd = os.path.abspath(os.path.dirname(__file__))
        self.checkpoint_path = os.path.join(cwd, "checkpoints")
        self.summary_path = os.path.join(cwd, "runs")
        self.dataset = RandAugmentationDataSet(path=os.path.join(cwd, "images"), origin_dir="origin", reduced_dir="reduced", limit=640000)
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
        self.tensorboard_writer = SummaryWriter(os.path.join(self.summary_path, f"srnet_trainer_{datetime.datetime.now():%Y%m%d%_H%M%S}"))

    def save_checkpoints(self, epoch, loss_val) -> None:
        checkpoint = {
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "schedular_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "loss": self.loss_fn.state_dict(),
        }
        filename = f"checkpoint_epoch{epoch}_loss{loss_val}"
        torch.save(checkpoint, os.path.join(self.checkpoint_path, filename))

    def load_checkpoints(self):
        checkpoint_list = os.listdir(self.checkpoint_path)
        if not checkpoint_list:
            return None
        created = 0
        last_checkpoint = None
        for filename in checkpoint_list:
            filepath = os.path.join(self.checkpoint_path, filename)
            if os.stat(filepath).st_mtime > created:
                last_checkpoint = filepath
        return torch.load(last_checkpoint)

    def _train(self):
        self.net.train()
        for batch_idx, (x, y) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()  # To reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
            x, y = x.to(self.device), y.to(self.device)
            pred = self.net(x)
            loss = self.loss_fn(pred, y)
            # BackPropagation
            loss.backward()
            self.optimizer.step()  # To adjust the parameters by the gradients collected in the backward pass.

            if batch_idx % 10 == 0:
                train_loss = loss.item()

    def _test(self):
        self.net.eval()

        test_loss, correct = 0, 0
        num_batches = len(self.test_dataloader)
        size = self.dataset.limit

        with torch.no_grad():
            for x, y in self.test_dataloader:
                pred = self.net(x)
                test_loss += self.loss_fn(pred, y).item()
                correct = 1 - test_loss

        test_loss /= num_batches
        correct /= size
        logger.info(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def train(self):
        for epoch in range(self.epochs):
            self._train()
            self._test()
            self.scheduler.step()



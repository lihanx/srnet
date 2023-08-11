# -*- coding:utf-8 -*-
import datetime
import os.path
import logging
logging.basicConfig(level=logging.INFO)

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from dataset import RandAugmentationDataSet
from models import SRNet
from loss import SSIMLoss
from utils import _check_dir


logger = logging.getLogger(__name__)


class SRNetTrainer:

    def __init__(self):
        self.epochs = 1500
        self.learning_rate = 1e-4
        self.batch_size = 64
        cwd = os.path.abspath(os.path.dirname(__file__))
        self.checkpoint_path = os.path.join(cwd, "checkpoints")
        _check_dir(self.checkpoint_path)
        self.summary_path = os.path.join(cwd, "runs")
        _check_dir(self.summary_path)
        self.weight_path = os.path.join(cwd, "weights")
        _check_dir(self.weight_path)
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
        self._training_date = datetime.datetime.now()
        self.summary_writer = SummaryWriter(os.path.join(self.summary_path, f"srnet_trainer_{self._training_date:%Y%m%d%_H%M%S}"))

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

    def _train(self, epoch):
        self.net.train()
        running_loss = 0.
        last_loss = 0.
        for batch_idx, (x, y) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()  # To reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
            x, y = x.to(self.device), y.to(self.device)
            pred = self.net(x)
            loss = self.loss_fn(pred, y)
            # BackPropagation
            loss.backward()
            self.optimizer.step()  # To adjust the parameters by the gradients collected in the backward pass.

            running_loss += loss.item()
            if batch_idx % 10 == 0:
                last_loss = running_loss / 10
                logger.info(f"Train batch {batch_idx + 1} loss: {last_loss}")
                tb_x = epoch * len(self.train_dataloader) + batch_idx + 1
                self.summary_writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.
        return last_loss

    def _test(self, epoch):
        self.net.eval()
        test_loss = 0.
        last_loss = 0.
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.test_dataloader):
                pred = self.net(x)
                test_loss += self.loss_fn(pred, y).item()
                if batch_idx % 10 == 0:
                    last_loss = test_loss / 10
                    logger.info(f"Validate batch {batch_idx + 1} loss: {last_loss}")
                    tb_x = epoch * len(self.test_dataloader) + batch_idx + 1
                    self.summary_writer.add_scalar("Loss/validate", last_loss, tb_x)
                    test_loss = 0.
        return last_loss

    def train(self):
        logger.info("Train start.")
        limit = 0.97
        best_ssim = 0.
        for epoch in range(self.epochs):
            logger.info(f"Training Epoch {epoch}/{self.epochs}")
            tloss = self._train(epoch)
            vloss = self._test(epoch)
            self.summary_writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": tloss, "Validation": vloss},
                epoch+1
            )
            self.scheduler.step()
            if 1 - vloss >= 0.97:
                logger.info(f"SSIM >= {limit}, Stop Training.")
                best_ssim = 1 - vloss
                break
        torch.save(self.net, f"srnet_{self._training_date:%Y%m%d%H%M%S}_loss{best_ssim}_.pth")
        logger.info("Model saved.")
        logger.info("Done.")


if __name__ == '__main__':
    trainer = SRNetTrainer()
    trainer.train()
# -*- coding:utf-8 -*-

import datetime
import os.path
import logging
logging.basicConfig(level=logging.INFO)
import sys
from typing import Union
from argparse import ArgumentParser

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

    def __init__(self,
                 epoch: int = 10000,
                 batch_size: int = 16,
                 earlystop_at: float = 0.3,
                 checkpoint: Union[None, str] = None):
        self.epochs = epoch
        self.learning_rate = 1e-4
        self.batch_size = batch_size
        self.data_count = 1600
        self.earlystop_at = earlystop_at
        cwd = os.path.abspath(os.path.dirname(__file__))
        self.checkpoint_path = os.path.join(cwd, "checkpoints")
        _check_dir(self.checkpoint_path)
        self.summary_path = os.path.join(cwd, "runs")
        _check_dir(self.summary_path)
        self.weight_path = os.path.join(cwd, "weights")
        _check_dir(self.weight_path)
        self.train_dataset = RandAugmentationDataSet(path=os.path.join(cwd, "images"), origin_dir="origin", reduced_dir="reduced", limit=self.data_count)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size)
        self.test_dataset = RandAugmentationDataSet(path=os.path.join(cwd, "images"), origin_dir="origin", reduced_dir="reduced", limit=self.data_count)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = SRNet()
        self.net.to(self.device)
        self.optimizer = Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_fn = SSIMLoss()
        self.lr_decay_rate = 0.8
        self.lr_epoch_per_decay = 20
        self.scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: self.lr_decay_rate ** (epoch // self.lr_epoch_per_decay))
        self._training_date = datetime.datetime.now()
        self.last_epoch = 0
        if checkpoint is not None:
            self.load_checkpoints(checkpoint)
        else:
            self.summary_writer = SummaryWriter(os.path.join(self.summary_path, f"srnet_trainer_{self._training_date:%Y%m%d%H%M%S}"))

    def save_checkpoints(self, epoch, loss_val) -> None:
        checkpoint = {
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "schedular_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "loss_state_dict": self.loss_fn.state_dict(),
        }
        filename = f"checkpoint_{self._training_date:%Y%m%d%H%M%S}_epoch{epoch}_loss{loss_val:.4f}.ckpt"
        torch.save(checkpoint, os.path.join(self.checkpoint_path, filename))
        weight_file = filename.replace("ckpt", "pth")
        torch.save(self.net.state_dict(), os.path.join(self.weight_path, weight_file))
        logger.info(f"Checkpoint saved: {filename}")

    def load_checkpoints(self, checkpoint):
        checkpoint_path = os.path.join(self.checkpoint_path, checkpoint)
        ckpt = torch.load(checkpoint_path)
        self.net.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["schedular_state_dict"])
        self.last_epoch = ckpt["epoch"]
        self.loss_fn.load_state_dict(ckpt["loss_state_dict"])
        d = checkpoint.split("_")[1]
        self._training_date = datetime.datetime.strptime(d, "%Y%m%d%H%M%S")
        self.summary_writer = SummaryWriter(os.path.join(self.summary_path, f"srnet_trainer_{d}"))
        logger.info(f"Checkpoint loaded: {checkpoint}")
        return None

    def _train(self, epoch):
        self.net.train()
        running_loss = 0.
        last_loss = 0.
        for batch_idx, (x, y) in enumerate(self.train_dataloader, start=1):
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
                logger.info(f"Train batch {batch_idx} loss: {last_loss}")
                tb_x = epoch * len(self.train_dataloader) + batch_idx + 1
                self.summary_writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.
        return last_loss

    def _test(self, epoch):
        self.net.eval()
        test_loss = 0.
        last_loss = 0.
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.test_dataloader, start=1):
                x, y = x.to(self.device), y.to(self.device)
                pred = self.net(x)
                test_loss += self.loss_fn(pred, y).item()
                if batch_idx % 10 == 0:
                    last_loss = test_loss / 10
                    logger.info(f"Validate batch {batch_idx} loss: {last_loss}")
                    tb_x = epoch * len(self.test_dataloader) + batch_idx + 1
                    self.summary_writer.add_scalar("Loss/validate", last_loss, tb_x)
                    self.summary_writer.add_images("Image/Origin", x, tb_x)
                    self.summary_writer.add_images("Image/Target", y, tb_x)
                    self.summary_writer.add_images("Image/Inference", pred, tb_x)
                    test_loss = 0.
        return last_loss

    def train(self):
        logger.info("Train start.")
        limit = self.earlystop_at
        best_loss = 0.1
        for epoch in range(self.last_epoch+1, self.epochs+1):
            logger.info(f"Training Epoch {epoch}/{self.epochs}")
            tloss = self._train(epoch)
            vloss = self._test(epoch)
            logger.info(f"Training Epoch {epoch} finished at: tloss-{tloss} vloss-{vloss}")
            self.summary_writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": tloss, "Validation": vloss},
                epoch
            )
            self.scheduler.step()
            if vloss < best_loss:
                self.save_checkpoints(epoch, vloss)
                best_loss = vloss
            elif epoch % 10 == 0:
                self.save_checkpoints(epoch, vloss)
            if tloss <= limit and vloss <= limit:
                logger.info(f"SSIM >= {1-best_loss}, Stop Training.")
                break
        torch.save(self.net.state_dict(), os.path.join(self.weight_path, f"srnet_{self._training_date:%Y%m%d%H%M%S}_loss{best_loss}.pth"))
        logger.info("Model saved.")
        logger.info("Done.")


def parse_train_args(args):
    parser = ArgumentParser()
    parser.add_argument("--epoch", type=int, help="指定训练 epoch", default=10000)
    parser.add_argument("--batch_size", type=int, help="指定训练 batch_size", default=16)
    parser.add_argument("--earlystop_at", type=float, help="指定训练提前停止的阈值", default=0.05)
    parser.add_argument("--checkpoint", help="指定保存的断点名称，继续进行训练", default=None)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_train_args(sys.argv[1:])
    trainer = SRNetTrainer(epoch=args.epoch,
                           batch_size=args.batch_size,
                           earlystop_at=args.earlystop_at,
                           checkpoint=args.checkpoint)
    trainer.train()

# -*- coding:utf-8 -*-

import os
import logging

import torch

logger = logging.getLogger(__name__)


def _check_dir(dirpath):
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
        logger.info(f"Create directory {dirpath}.")


def _save_weight_from_checkpoint(ckpt, weight_path):
    torch.save(ckpt["model_state_dict"], weight_path)
    logger.info(f"Model weights saved: {weight_path}")
# -*- coding:utf-8 -*-

import os
import logging

logger = logging.getLogger(__name__)


def _check_dir(dirpath):
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
        logger.info(f"Create directory {dirpath}.")
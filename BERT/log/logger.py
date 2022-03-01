#
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import logging
from logging import handlers
import time


def logger_fn(name, input_file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.propagate = False  # 不将信息传递给祖先
    tf_logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # fh = logging.FileHandler(input_file, encoding="utf-8", mode='w')

    fh = handlers.TimedRotatingFileHandler(input_file, when="D",
                                           backupCount=7, interval=1, encoding='utf-8')
    formatter_fh = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter_fh)
    tf_logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.ERROR)  # 控制台仅仅输出
    formatter_sh = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter_sh)
    tf_logger.addHandler(sh)
    return tf_logger


def logger_debug_fn(name, input_file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.propagate = False  # 不将信息传递给祖先
    tf_logger.setLevel(logging.INFO)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(input_file, encoding="utf-8", mode='w')
    formatter_fh = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter_fh)
    tf_logger.addHandler(fh)

    return tf_logger

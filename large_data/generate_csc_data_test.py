# -*- coding: UTF-8 -*-

import sys
import os
import re
from optparse import OptionParser
from tqdm import tqdm
import pickle
import numpy as np
import re
from strTools import uniform, is_chinese
import pickle


def normalize_uniform(text):
    """
    # 非汉字全部半角化？
    # 统一起来会比较好
    :param text:
    :return:
    """
    text = uniform(text).lower()
    text = re.sub("\s+", "", text)

    return text


def normalize_lower(text):
    """
    # 非汉字全部半角化？
    # 统一起来会比较好
    :param text:
    :return:
    """
    text = text.lower()
    text = re.sub("\s+", "", text)
    return text


def normalize(text):
    """
    # 非汉字全部半角化？
    # 统一起来会比较好
    :param text:
    :return:
    """
    text = re.sub("\s+", "", text)
    return text


def process_path(path_in, path_out):
    with open(path_in, "r", encoding="utf-8") as f, open(path_out, "w", encoding="utf-8") as fw:
        for line in f.readlines():
            src, trg = line.strip().split(" ")
            src, trg = normalize_lower(src), normalize_lower(trg)
            fw.write(src + " " + trg + "\n")


if __name__ == "__main__":
    path_in = "/data_local/TwoWaysToImproveCSC/BERT/data/13train.txt"
    path_out = "/data_local/TwoWaysToImproveCSC/BERT/data/13train_lower.txt"
    process_path(path_in, path_out)

    path_in = "/data_local/TwoWaysToImproveCSC/BERT/data/13test.txt"
    path_out = "/data_local/TwoWaysToImproveCSC/BERT/data/13test_lower.txt"
    process_path(path_in, path_out)

    path_in = "/data_local/TwoWaysToImproveCSC/BERT/cc_data/chinese_spell_4.txt"
    path_out = "/data_local/TwoWaysToImproveCSC/BERT/cc_data/chinese_spell_lower_4.txt"
    process_path(path_in, path_out)

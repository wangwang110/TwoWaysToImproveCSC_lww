#
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys


def remove_correct_sent(trains, path_out):
    """
    统计错误模式
    :param train:
    :return:
    """
    with open(path_out, "w", encoding="utf-8") as fw:
        for train in trains:
            with open(train, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    try:
                        src, trg = line.strip().split()
                    except Exception as e:
                        print(e)
                        continue
                    if src == trg:
                        continue
                    else:
                        fw.write(src + " " + trg + "\n")


pretrain_path = [
    "/data_local/chinese_data/original/new_pretrain_sep_2.train",
    "/data_local/chinese_data/original/new_pretrain_sep_1.train",
    "/data_local/chinese_data/original/new_pretrain_sep_0.train"
]
pretrain_path_out = "/data_local/chinese_data/original/new_pretrain_sep_all_witherror.train"
remove_correct_sent(pretrain_path, pretrain_path_out)

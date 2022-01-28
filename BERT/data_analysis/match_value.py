# Copyright 2018 CVTE . All Rights Reserved.
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


def count(trains):
    """
    统计错误模式
    :param train:
    :return:
    """
    train_set = set()
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
                for s, t in zip(list(src), list(trg)):
                    if s != t:
                        train_set.add((s, t))
    return train_set


def calc_val(train, test):
    train_set = count(train)
    test_set = count(test)

    return len(train_set & test_set) / len(test_set)


#
train = [
    "/data_local/TwoWaysToImproveCSC/BERT/data/pretrain_auto.train",
    "/data_local/TwoWaysToImproveCSC/BERT/data/rep_autog_wang_train.txt",
    "/data_local/TwoWaysToImproveCSC/BERT/data/13train.txt"
]
test = ["/data_local/TwoWaysToImproveCSC/BERT/data/13test.txt"]
print(calc_val(train, test))
# 0.969(0.968,0.14)
# 加上预训练的数据
# 0.977

# train = [
#     "/data_local/TwoWaysToImproveCSC/BERT/data/rep_autog_wang_train.txt",
#     "/data_local/TwoWaysToImproveCSC/BERT/data/14train.txt"
# ]
# test = ["/data_local/TwoWaysToImproveCSC/BERT/data/14test.txt"]
# print(calc_val(train, test))
# # 0.971
#
# train = [
#     "/data_local/TwoWaysToImproveCSC/BERT/data/rep_autog_wang_train.txt",
#     "/data_local/TwoWaysToImproveCSC/BERT/data/15train.txt"
# ]
# test = ["/data_local/TwoWaysToImproveCSC/BERT/data/15test.txt"]
# print(calc_val(train, test))
# # 0.974


# 作文测试集
train = [
    "/data_local/TwoWaysToImproveCSC/BERT/data/pretrain_auto.train",
    "/data_local/TwoWaysToImproveCSC/BERT/data/rep_autog_wang_train.txt",
    "/data_local/TwoWaysToImproveCSC/BERT/data/13train.txt"
]
# 0.619 (0.066 + 0.619) 13
# 0.634 14
# 0.628 15
test = ["/data_local/TwoWaysToImproveCSC/BERT/cc_data/chinese_spell_4.txt"]
print(calc_val(train, test))
# 加上预训练的数据，确定是否需要扩充候选集
# 有这样的模式，不一定有这样的上下文
# 0.801

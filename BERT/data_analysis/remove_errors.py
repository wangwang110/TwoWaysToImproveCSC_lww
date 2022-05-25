#
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys


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
    print(test_set - train_set)

    return len(train_set & test_set) / len(test_set)


# test = ["/data_local/TwoWaysToImproveCSC/BERT/data/13test.txt"]
# test = ["/data_local/TwoWaysToImproveCSC/BERT/data/chinese_spell_4.txt"]

test = [sys.argv[1]]
test_set = count(test)

# pretrain_path = "/data_local/chinese_data/original/new_pretrain_all.train"
# pretrain_path_out = "/data_local/chinese_data/original/new_pretrain_remove_cc.train"

pretrain_path = sys.argv[2]
pretrain_path_out = sys.argv[3]

with open(pretrain_path, "r", encoding="utf-8") as f, open(pretrain_path_out, "w", encoding="utf-8") as fw:
    for line in f.readlines():
        src, trg = line.strip().split(" ")
        src_tokens = list(src)
        trg_tokens = list(trg)
        i = 0
        for s, t in zip(src_tokens, trg_tokens):
            if s != t and (s, t) not in test_set:
                src_tokens[i] = t
            i += 1
        fw.write("".join(src_tokens) + " " + trg + "\n")

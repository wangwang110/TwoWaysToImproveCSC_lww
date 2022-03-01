#
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import sys
import random

"""
增加候选集之后，重新构造数据，整个数据的变化很大。 这个时候结果难以比较

那么用不同来源的数据不是更加难以比较？？

"""
# 完全符合作文的混淆集
confusion_set = {}
with open('/data_local/TwoWaysToImproveCSC/BERT/save/cc_confuse.txt', "r") as f:
    text = f.read().strip().split("\n\n")
    for item_str in text:
        item_li = item_str.strip().split("\n")
        s, t = item_li[0].strip(), item_li[1].strip()
        print(s, t)
        if s in confusion_set:
            confusion_set[s].append(t)
        else:
            confusion_set[s] = []
            confusion_set[s].append(t)

path = "../data/wiki_00_base.train"
path_out = "../data/wiki_00_base_confuse.train"
all_dict = {}
with open(path, "r", encoding="utf-8") as f, open(path_out, "w", encoding="utf-8") as fw:
    for line in f.readlines():
        src, trg = line.strip().split(" ")
        tokens = []
        if src == trg:
            # 原来构造数据时，咩有修改
            for s in list(src):
                if s in confusion_set:
                    tokens.append(confusion_set[s][0])
                else:
                    tokens.append(s)
        else:
            for s in list(src):
                if s in confusion_set and random.random() < 0.5:
                    tokens.append(confusion_set[s][0])
                else:
                    tokens.append(s)
            #
            # new_tokens = []
            # if "".join(tokens) != src:
            #     for s, t in zip(list(src), list(trg)):
            #         if s != t and s not in confusion_set:

        fw.write("".join(tokens) + " " + trg)

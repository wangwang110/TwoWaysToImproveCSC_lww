# Copyright 2018 CVTE . All Rights Reserved.
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import sys

sys.path.append("..")
from model import li_testconstruct, BertDataset, BFTLogitGen, readAllConfusionSet, cc_testconstruct, construct

path = "./spellGraphs.txt"
path_out = "./spellGraphs_confusion.txt"
all_dict = {}
with open(path, "r", encoding="utf-8") as f, open(path_out, "w", encoding="utf-8") as fw:
    for line in f.readlines():
        s, t, r = line.strip().split("|")
        if r not in ["同音同调", "同音异调", "形近"]:
            continue
        if s not in all_dict:
            all_dict[s] = set([t])
        else:
            all_dict[s].add(t)

        if t not in all_dict:
            all_dict[t] = set([s])
        else:
            all_dict[t].add(s)

    for key in all_dict:
        fw.write(key + "==\n")
        fw.write(" ".join(all_dict[key]) + "\n")
        fw.write("\n")

    confusion_set = readAllConfusionSet('/data_local/TwoWaysToImproveCSC/BERT/save/confusion.file')
    # for key in confusion_set:
    #     if key not in all_dict:
    #         print(key)
    #         print(confusion_set[key])

    for key in all_dict:
        if key not in confusion_set:
            print(key)
            print(all_dict[key])

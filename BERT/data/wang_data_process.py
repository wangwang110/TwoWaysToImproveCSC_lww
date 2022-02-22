# -*- coding: UTF-8 -*-

import sys
import os
import re
import pickle

vob = set()
with open("/data_local/plm_models/chinese_L-12_H-768_A-12/vocab.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        vob.add(line.strip())


# def normalize_uniform(text):
#     """
#     # 非汉字全部半角化？
#     # 统一起来会比较好
#     :param text:
#     :return:
#     """
#     text = uniform(text).lower()
#     text = re.sub("\s+", "", text)
#
#     return text


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


# def normalize(text):
#     """
#     # 非汉字全部半角化？
#     # 统一起来会比较好
#     :param text:
#     :return:
#     """
#     text = re.sub("\s+", "", text)
#     return text


def readAllConfusionSet(filepath):
    with open(filepath, 'rb') as f:
        allSim = pickle.load(f)
        return allSim


def process_wang(path):
    path_out = os.path.dirname(path) + "/unk_" + "".join(os.path.basename(path).split("_")[1:])
    with open(path, "r", encoding="utf-8") as f, open(path_out, "w", encoding="utf-8") as fw:
        for line in f.readlines():
            src, trg = line.strip().split(" ")
            src = normalize_lower(src)
            src_tokens = list(src)
            trg = normalize_lower(trg)
            trg_tokens = list(trg)
            i = 0
            for s, t in zip(src_tokens, trg_tokens):
                if s not in vob and t in vob:
                    src_tokens[i] = trg_tokens[i]
                    print(s)
                    print(t)
                if s in vob and t not in vob:
                    src_tokens[i] = trg_tokens[i]
                    print(s)
                    print(t)
                i += 1

            src = "".join(src_tokens)
            trg = "".join(trg_tokens)
            fw.write(src + " " + trg + "\n")


if __name__ == "__main__":
    path = "/data_local/TwoWaysToImproveCSC/BERT/data/rep_autog_wang_train.txt"
    process_wang(path)

    path = "/data_local/TwoWaysToImproveCSC/BERT/data/rep_autog_wang_1k_dev.txt"
    process_wang(path)

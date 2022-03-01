#
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

path1 = "./xiaoxue_sent_all.txt"
path2 = "./xiaoxue_sent_all_cor.txt"
path_out = "./xiaoxue_sent_with_error.txt"

with open(path1, "r", encoding="utf-8") as f1, open(path2, "r", encoding="utf-8") as f2, \
        open(path_out, "w", encoding="utf-8") as fw:
    all_texts = []
    for line in f1.readlines():
        key, src = line.strip().split(" ")
        all_texts.append(src)

    all_texts_correct = set()
    for line in f2.readlines():
        src = line.strip()
        all_texts_correct.add(src)

    for src in all_texts:
        if src not in all_texts_correct:
            fw.write(src + "\n")

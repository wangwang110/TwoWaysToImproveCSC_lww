# Copyright 2018 CVTE . All Rights Reserved.
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pickle

path = "/data_local/TwoWaysToImproveCSC/large_data/weibo/weibo_vocab.pkl"
weibo_vocab_dict = pickle.load(open(path, "rb"))

path = "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_vocab.pkl"
wiki_vocab_dict = pickle.load(open(path, "rb"))
wiki_vocab = sorted(wiki_vocab_dict.items(), key=lambda s: int(s[1]), reverse=True)

num = 3500
with open("./all_" + str(num) + "_chars.txt", "w", encoding="utf-8") as fw:
    i = 0
    for item in wiki_vocab:
        word, fre = item
        fw.write(word + "\n")
        i += 1
        if i == num:
            break
#
print(len(weibo_vocab_dict))
print(len(wiki_vocab_dict))

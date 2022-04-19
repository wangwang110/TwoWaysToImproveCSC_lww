# -*- coding: UTF-8 -*-

import sys
import os
import re
from optparse import OptionParser
from tqdm import tqdm
import json
import re
import pickle


def cut_sent(para):
    """规则分句"""
    para = re.sub("([。！？!\?]){2,}", r"\1", para)
    para = re.sub("<br><br>", "\n", para)
    para = re.sub('([。！？!\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    #
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？!\?][”’])([^，。！？!\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


def refine_vocab():
    """
    获取一个更简化的字典
    每个汉字都在bert的词表中
    目的是将bert词表中的一些非汉字去掉
    用keep作为标签
    :return:
    精简词表
    """
    bert_vocab_path = "/data_local/plm_models/chinese_L-12_H-768_A-12/vocab.txt"
    vob = {}
    with open(bert_vocab_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            vob.setdefault(line.strip(), i)

    wiki_vocab_path = "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_vocab.pkl"
    wiki_vocab = pickle.load(open(wiki_vocab_path, "rb"))

    weibo_vocab_path = "/data_local/TwoWaysToImproveCSC/large_data/weibo/weibo_vocab.pkl"
    weibo_vocab = pickle.load(open(weibo_vocab_path, "rb"))

    character = {"[PAD]": 100, "[UNK]": 100, "[CLS]": 100, "[SEP]": 100, "[MASK]": 100}
    for s in weibo_vocab:
        if s not in character:
            character[s] = weibo_vocab[s]
        else:
            character[s] += weibo_vocab[s]

    for s in wiki_vocab:
        if s not in character:
            character[s] = wiki_vocab[s]
        else:
            character[s] += wiki_vocab[s]

    new_vocab = {}
    for s in character:
        if character[s] <= 20:
            continue
        if s in vob:
            new_vocab[s] = vob[s]
            print(s)
    print(len(new_vocab))
    pickle.dump(new_vocab, open("./bert_vocab_refine.pkl", "wb"), protocol=0)

    # sorted_dict = sorted(character.items(), key=lambda s: s[1], reverse=True)
    # with open("./vocab.txt", "w", encoding="utf-8") as fw:
    #     for item in sorted_dict:
    #         fw.write(item[0] + "\t" + str(item[1]) + "\n")
    #         print(item)
    # print(len(sorted_dict))

#
# # 引号里面有好多句子，怎么办
# res = cut_sent('爱迪生曾说过:“天才是百分只99%的汗水，加百分之一努力。天才是百分只99%的汗水，加百分之一努力。”，所以我希望你不要放弃。')
# print(res)
# refine_vocab()

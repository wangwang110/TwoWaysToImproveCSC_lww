# -*- coding: utf-8 -*-

import time
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import json
import re
import spacy
import os
import pickle
from tqdm import tqdm
import sys
from pypinyin import pinyin, lazy_pinyin, Style

if __name__ == '__main__':
    paths = [
        # "/data_local/TwoWaysToImproveCSC/large_data/tmp.txt"
        "/data_local/TwoWaysToImproveCSC/large_data/new2016zh/news2016zh_cor.txt",
        "/data_local/TwoWaysToImproveCSC/large_data/webtext2019zh/web_zh_correct.txt",
        "/data_local/TwoWaysToImproveCSC/large_data/translation2019zh/translation2019zh_correct.txt",
        "/data_local/TwoWaysToImproveCSC/BERT/cc_data/xiaoxue_sent_all_cor.txt",
        "/data_local/TwoWaysToImproveCSC/large_data/weibo/weibo_correct_first.txt",
        "/data_local/TwoWaysToImproveCSC/large_data/weibo/weibo_correct_second.txt",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_00_sent",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_01_sent",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_02_sent",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_03_sent",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_04_sent",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_05_sent",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_06_sent",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_07_sent",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_08_sent",
    ]
    if os.path.exists("./vocab_zuowen.pkl"):
        zh_ci_dict = pickle.load(open("./vocab_zuowen.pkl", "rb"))
    else:
        zh_ci_dict = {}
        nlp = spacy.load("zh_core_web_sm")
        for path in paths:
            print(path)
            path_out = "".join(path.split(".")[:-1]) + ".tok"
            with open(path, "r", encoding="utf-8") as f, open(path_out, "w", encoding="utf-8") as fw:
                texts = [sent.strip() for sent in f.readlines()]
                num = len(texts)
                print(num)
                i = 0
                for doc in nlp.pipe(texts, disable=["tok2vec", "tagger", "parser", "ner"], n_process=56):
                    i += 1
                    words = [item.text for item in doc]
                    fw.write(" ".join(words) + "\n")
                    for word in words:
                        if word not in zh_ci_dict:
                            zh_ci_dict[word] = 1
                        else:
                            zh_ci_dict[word] += 1
                    if i % 10000 == 0:
                        print("========{}============".format(round(i / num, 2)))

        pickle.dump(zh_ci_dict, open("./vocab_zuowen.pkl", "wb"))

new_zh_ci = {}
sort_zh_ci_dict = sorted(zh_ci_dict.items(), key=lambda s: s[1], reverse=True)
for item in sort_zh_ci_dict:
    word, fre = item
    if 1 < len(word) < 4 and not re.search('[^\u4e00-\u9fa5]', word) and fre > 10:  # 必须是汉字
        new_zh_ci[word] = fre

print(len(new_zh_ci))
print(new_zh_ci["逆境"])
print(new_zh_ci["一旦"])
print(new_zh_ci["关卡"])
print(new_zh_ci["白费"])
pickle.dump(new_zh_ci, open("./vocab_xiaoxue.pkl", "wb"))
print(len(new_zh_ci))

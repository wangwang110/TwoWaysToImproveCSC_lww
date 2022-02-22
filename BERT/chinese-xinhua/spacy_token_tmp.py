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
    nlp = spacy.load("zh_core_web_sm")
    dir_path = "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent_tok/"
    for path in paths:
        print(path)
        path_out = dir_path + os.path.basename(path) + ".tok"
        with open(path, "r", encoding="utf-8") as f, open(path_out, "w", encoding="utf-8") as fw:
            texts = [sent.strip() for sent in f.readlines()]
            num = len(texts)
            print(num)
            i = 0
            for doc in nlp.pipe(texts, disable=["tok2vec", "tagger", "parser", "ner"], n_process=56):
                i += 1
                words = [item.text for item in doc]
                fw.write(" ".join(words) + "\n")
                if i % 10000 == 0:
                    print("========{}============".format(round(i / num, 2)))

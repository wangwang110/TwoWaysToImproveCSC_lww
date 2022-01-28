# -*- coding: utf-8 -*-
# get_similiar_char.py
import numpy as np
from pypinyin import pinyin, lazy_pinyin, Style
import os
import pickle


# 获取所有汉字的向量表示，以dict储存
def get_all_char_pinyin():
    path = "/data_local/TwoWaysToImproveCSC/large_data/all_3500_chars.txt"
    pinyin_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            ch = line.strip()
            ch_pinyin = pinyin(ch, style=Style.TONE3, heteronym=True)
            for p_li in ch_pinyin:
                for p in p_li:
                    if p not in pinyin_dict:
                        pinyin_dict[p] = [ch]
                    else:
                        pinyin_dict[p].append(ch)
    return pinyin_dict


if __name__ == '__main__':
    pinyin_dict = get_all_char_pinyin()

    # 获取同音汉字
    similarity_dict = {}
    match_char = "在"
    ch_pinyin = pinyin(match_char, style=Style.TONE3, heteronym=True)
    res = []
    for p_li in ch_pinyin:
        for p in p_li:
            print(p)
            if match_char in pinyin_dict[p]:
                pinyin_dict[p].remove(match_char)
            res.extend(pinyin_dict[p])
    print(res)

    # # 按相似度排序，取前10个
    # sorted_similarity = sorted(similarity_dict.items(), key=itemgetter(1), reverse=True)
    # print([(char, round(similarity, 4)) for char, similarity in sorted_similarity[:10]])

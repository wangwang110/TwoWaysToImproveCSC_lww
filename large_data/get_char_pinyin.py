# -*- coding: utf-8 -*-
import numpy as np
from pypinyin import pinyin, lazy_pinyin, Style
import os
import pickle


# 获取所有汉字的拼音表示，相同拼音的存储在以拼音为key的字典中，考虑声调
def get_all_char_pinyin():
    path = "/data_local/TwoWaysToImproveCSC/large_data/all_3500_chars.txt"
    pinyin_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            ch = line.strip()
            ch_pinyin = pinyin(ch, style=Style.TONE3, heteronym=False)
            # heteronym 是否启用多音字模式
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
    match_char = "明"
    ch_pinyin = pinyin(match_char, style=Style.TONE3, heteronym=False)
    res = []
    for p_li in ch_pinyin:
        for p in p_li:
            print(p)
            if match_char in pinyin_dict[p]:
                pinyin_dict[p].remove(match_char)
            res.extend(pinyin_dict[p])
    print(res)


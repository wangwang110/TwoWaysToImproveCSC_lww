# coding: utf-8

"""
@File    : get_multi_pinyin_char.py.py
@Time    : 2022/5/19 15:09
@Author  : liuwangwang
@Software: PyCharm
"""
import os
import re
import pickle
from pypinyin import pinyin, lazy_pinyin, Style

zh_vocab_path = "./data/vocab_word.pkl"
new_zh_ci = pickle.load(open(zh_vocab_path, "rb"))

# 获取多音字
multi_pinyin_chars = {}

for key in new_zh_ci:
    for s in key:
        w_pinyin_li = pinyin(key, style=Style.TONE3, heteronym=True)
        pinyin_li = set()
        for s1 in w_pinyin_li[0]:
            s1 = re.sub("\d$", "", s1)
            pinyin_li.add(s1)
        if len(pinyin_li) > 1:
            multi_pinyin_chars[s] = pinyin_li
#
print(multi_pinyin_chars)

print(len(multi_pinyin_chars))

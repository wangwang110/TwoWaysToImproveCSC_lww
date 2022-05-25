# -*- coding: utf-8 -*-

import re
import pickle
from pypinyin import pinyin, lazy_pinyin, Style
from get_similar_char_edit import CharFuncs

"""
扩大混淆集，使得混淆集覆盖度更高
"""

if __name__ == '__main__':

    c = CharFuncs('./data/char_meta_new.txt')
    # 计算字形相似度

    zh_ci_dict = pickle.load(open("./data/vocab_zuowen.pkl", "rb"))
    # 词及其词频

    confset_path = "/data_local/TwoWaysToImproveCSC/BERT/save/confusion.file"
    confuse_set = pickle.load(open(confset_path, "rb"))
    # 原有混淆集

    same_pinyin_words = pickle.load(open("./data/same_pinyin_words.pkl", "rb"))
    same_pinyin_chars = pickle.load(open("./data/same_pinyin_chars.pkl", "rb"))
    # 同音字

    # 形近字
    similar_shape_char = {}
    similar_shape_paths = ["/data_local/TwoWaysToImproveCSC/large_data/tmp0.8.txt",
                           "/data_local/TwoWaysToImproveCSC/large_data/tmp0.96.txt"]
    for pth in similar_shape_paths:
        with open(pth, "r", encoding="utf-8") as f:
            for line in f.readlines():
                s, item = line.strip().split("\t")
                t = item.split(":")[0]
                if s not in similar_shape_char:
                    similar_shape_char[s] = set()
                similar_shape_char[s].add(t)

                if t not in similar_shape_char:
                    similar_shape_char[t] = set()
                similar_shape_char[t].add(s)

    # 将形近字和音近字加入原来的混淆集，但是没有增加key
    new_confuse_set = {}
    for s in confuse_set:
        if s not in new_confuse_set:
            new_confuse_set[s] = confuse_set[s]
        # 同拼音的加入
        w_pinyin = lazy_pinyin(s)
        w_pinyin_str = "_".join(w_pinyin)
        if w_pinyin_str in same_pinyin_chars:
            for t in same_pinyin_chars[w_pinyin_str]:
                new_confuse_set[s].add(t)
        # 形近字加入
        if s in similar_shape_char:
            for t in similar_shape_char[s]:
                new_confuse_set[s].add(t)

    for w_pinyin_str in same_pinyin_chars:
        for s in same_pinyin_chars[w_pinyin_str]:
            if s not in new_confuse_set:
                new_confuse_set[s] = set()
                for t in same_pinyin_chars[w_pinyin_str]:
                    if t != s:
                        new_confuse_set[s].add(t)

    for s in similar_shape_char:
        if s not in new_confuse_set:
            new_confuse_set[s] = set()
        for t in similar_shape_char[s]:
            if t != s:
                new_confuse_set[s].add(t)

    pickle.dump(new_confuse_set, open("./data/large_confuse_set_new.pkl", "wb"), protocol=0)

# 考虑模糊音 模糊音 （平翘舌，前鼻音后鼻音）
# https://baike.baidu.com/item/%E6%A8%A1%E7%B3%8A%E9%9F%B3/4127723
# 用微信数据+作文数据的分词
# 考虑模糊音
# 用faspell的字形相似度计算


# 我的预训练问题很大
# 但是应该不是混淆集覆盖度不够的问题（因为sighan13覆盖度达到97%,依然召回率不高）
# 但是有可能是混淆集太杂的问题，引入很多不常见的错误

# 可能是： 错误引入方式不对：
# 1.不能随机位置替换，应该每隔开多少汉字替换
# 2.错误引入的比例不对，形近，音近，字形的占比
# 3. 语言风格问题，来源问题

# 统计预训练数据集中，覆盖测试集和不覆盖测试集的占比
# 语法纠错数据，筛选出来的600w

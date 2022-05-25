# -*- coding: utf-8 -*-

import re
import pickle
from pypinyin import pinyin, lazy_pinyin, Style
from get_similar_char_edit import CharFuncs

"""
减小混淆集，使得混淆集更为合理
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

    new_confuse_set = {}
    for s in confuse_set:
        if s not in new_confuse_set:
            new_confuse_set[s] = set()

        print("\n\n======================")
        print("char:" + s)
        if s in zh_ci_dict:
            print("fre:" + str(zh_ci_dict[s]))
        else:
            print("fre:" + str(0))
        print("原有混淆集：\n" + " ".join(confuse_set[s]) + "\n")

        w_pinyin = lazy_pinyin(s)
        w_pinyin_str = "_".join(w_pinyin)
        if w_pinyin_str in same_pinyin_chars:
            print("音近：\n" + " ".join(same_pinyin_chars[w_pinyin_str]) + "\n")
            in_conf_set = confuse_set[s] & same_pinyin_chars[w_pinyin_str]
            print("在混淆集的：\n" + " ".join(in_conf_set) + "\n")
            for t in in_conf_set:
                new_confuse_set[s].add(t)

            not_conf_set = confuse_set[s] - same_pinyin_chars[w_pinyin_str]
            print("不在混淆集的：\n" + " ".join(not_conf_set) + "\n")

            shape_dict = {}
            for xx in not_conf_set:
                if xx == s:
                    continue
                # 拼音相同,上面的字典没统计到
                if lazy_pinyin(xx) == lazy_pinyin(s):
                    new_confuse_set[s].add(xx)

                elif xx not in shape_dict:
                    shape_dict[xx] = round(c.shape_similarity(xx, s), 3)

                # # 考虑模糊音
                # fuzzy_dict_final = {"ang": "an", "an": "ang", "eng": "en", "en": "eng", "ing": "in", "in": "ing",
                #                     "uang": "uan", "uan": "uang"}
                # fuzzy_dict_init = {"z": "zh", "zh": "z", "c": "ch", "ch": "c", "s": "sh", "sh": "s", "l": ["r", "n"],
                #                    "r": "l", "n": "l", "f": "h", "h": "f", }
                #
                # base = lazy_pinyin(s)[0]
                # base_li = [base]
                # for key in fuzzy_dict_final:
                #     base_final = base
                #     val = fuzzy_dict_final[key]
                #     if base_final.endswith(key):
                #         base_final = re.sub(key, val, base_final)
                #         base_li.append(base_final)
                #
                # for key in fuzzy_dict_init:
                #     base_init = base
                #     val = fuzzy_dict_init[key]
                #     if base_init.startswith(key):
                #         if type(val) == str:
                #             base_init = re.sub(key, val, base_init)
                #             base_li.append(base_init)
                #         else:
                #             for item in val:
                #                 base_init = re.sub(key, item, base_init)
                #                 base_li.append(base_init)
                #
                # # 考虑多音字
                # if len(set(pinyin(xx, heteronym=True)[0]) & set(pinyin(s, heteronym=True)[0])) != 0:
                #     print(xx)
                #     new_confuse_set[s].add(xx)
                # elif lazy_pinyin(xx)[0] in base_li:
                #     print(xx)
                #     new_confuse_set[s].add(xx)
                # else:
                #     if xx not in shape_dict:
                #         shape_dict[xx] = round(c.shape_similarity(xx, s), 3)

            sort_shape_dict = sorted(shape_dict.items(), key=lambda s: s[1], reverse=True)
            for item in sort_shape_dict:
                print(item)
                if item[1] > 0.333:
                    new_confuse_set[s].add(item[0])

    pickle.dump(new_confuse_set, open("./data/small_confuse_set_new.pkl", "wb"), protocol=0)

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

# -*- coding: utf-8 -*-

import re
import spacy
import os
import pickle
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
    if os.path.exists("./data/vocab_zuowen.pkl"):
        zh_ci_dict = pickle.load(open("./data/vocab_zuowen.pkl", "rb"))
    else:
        zh_ci_dict = {}
        nlp = spacy.load("zh_core_web_sm")
        for path in paths:
            path_out = "".join(path.split(".")[:-1]) + ".tok"
            with open(path, "r", encoding="utf-8") as f, open(path_out, "w", encoding="utf-8") as fw:
                texts = [sent.strip() for sent in f.readlines()]
                num = len(texts)
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

        pickle.dump(zh_ci_dict, open("./data/vocab_zuowen.pkl", "wb"))

zh_vocab_path = "./data/vocab_word.pkl"
if os.path.exists(zh_vocab_path):
    new_zh_ci = pickle.load(open(zh_vocab_path, "rb"))
else:
    new_zh_ci = {}
    sort_zh_ci_dict = sorted(zh_ci_dict.items(), key=lambda s: s[1], reverse=True)
    for item in sort_zh_ci_dict:
        word, fre = item
        if not re.search('[^\u4e00-\u9fa5]', word):  # 必须是汉字
            new_zh_ci[word] = fre
    pickle.dump(new_zh_ci, open(zh_vocab_path, "wb"))

print(len(new_zh_ci))
print(new_zh_ci["逆境"])
print(new_zh_ci["一旦"])
print(new_zh_ci["关卡"])
print(new_zh_ci["白费"])

# 获取同音词和同音字
same_pinyin_words = {}
same_pinyin_chars = {}

# 考虑模糊音
fuzzy_dict_final = {"ang": "an", "an": "ang", "eng": "en", "en": "eng", "ing": "in", "in": "ing",
                    "uang": "uan", "uan": "uang", "uo": "ou", "ie": "ei"}
fuzzy_dict_init = {"zh": "z", "z": "zh", "ch": "c", "c": "ch", "sh": "s", "s": "sh", "l": ["r", "n"],
                   "r": "l", "n": "l", "f": "h", "h": "f", }

# pinyin包多音字太多错误了, 不用他的
multi_char = {"乐": ["yue", "le"], "重": ["chong", "zhong"], "降": ["jiang", "xiang"], "落": ["luo", "la"],
              "了": ["liao", "le"], '的': ['di', 'de'], '地': ['di', 'de'], "得": ["de", "dei"],
              '曾': ['ceng', 'zeng'], '朝': ['chao', 'zhao']
              }

for key in new_zh_ci:
    if new_zh_ci[key] < 20:
        continue
    if len(key) >= 2:
        # 同音词
        w_pinyin_li = lazy_pinyin(key)
        w_pinyin_str = "_".join(w_pinyin_li)
        if w_pinyin_str not in same_pinyin_words:
            same_pinyin_words[w_pinyin_str] = set()
        same_pinyin_words[w_pinyin_str].add(key)

    # if len(key) == 2:
    #     # 两个字同音词
    #     # 考虑多音字
    #     w_pinyin_li = pinyin(key, style=Style.TONE3, heteronym=True)
    #     for s1 in w_pinyin_li[0]:
    #         for s2 in w_pinyin_li[1]:
    #             s1 = re.sub("\d$", "", s1)
    #             s2 = re.sub("\d$", "", s2)
    #             w_pinyin_str = s1 + "_" + s2
    #             if w_pinyin_str not in same_pinyin_words:
    #                 same_pinyin_words[w_pinyin_str] = set()
    #             same_pinyin_words[w_pinyin_str].add(key)

    for s in key:
        base = lazy_pinyin(s)[0]
        base_li = {base}
        # 考虑模糊音
        for k in fuzzy_dict_final:
            if base.endswith(k):
                val = fuzzy_dict_final[k]
                base_final = base[:-len(k)] + val
                base_li.add(base_final)
                break

        for k in fuzzy_dict_init:
            if base.startswith(k):
                val = fuzzy_dict_init[k]
                if type(val) == str:
                    base_init = val + base[len(k):]
                    base_li.add(base_init)
                    break
                else:
                    for item in val:
                        base_init = item + base[len(k):]
                        base_li.add(base_init)
                    break

        # 考虑多音字
        if s in multi_char:
            for xx in multi_char[s]:
                if xx != base:
                    base_li.add(xx)
        for c_pinyin_str in base_li:
            if c_pinyin_str not in same_pinyin_chars:
                same_pinyin_chars[c_pinyin_str] = set()
            same_pinyin_chars[c_pinyin_str].add(s)

print("end")
pickle.dump(same_pinyin_words, open("./data/same_pinyin_words.pkl", "wb"), protocol=0)
pickle.dump(same_pinyin_chars, open("./data/same_pinyin_chars.pkl", "wb"), protocol=0)

# 考虑模糊音 模糊音 （平翘舌，前鼻音后鼻音）
# https://baike.baidu.com/item/%E6%A8%A1%E7%B3%8A%E9%9F%B3/4127723
# 用微信数据+作文数据的分词
# 考虑模糊音
# 用faspell的字形相似度计算


# 我的预训练问题很大
# 但是应该不是混淆集覆盖度不够的问题（因为sighan13覆盖度达到97%）但是依然召回率不高
# 但是有可能是混淆集太杂的问题，引入很多不常见的错误

# 可能是： 错误引入方式不对：
# 1.不能随机位置替换，应该每隔开多少汉字替换
# 2.错误引入的比例不对，形近，音近，字形的占比
# 3. 语言风格问题，来源问题

# 统计预训练数据集中，覆盖测试集和不覆盖测试集的占比
# 语法纠错数据，筛选出来的600w

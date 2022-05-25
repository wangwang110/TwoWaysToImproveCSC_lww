# -*- coding: utf-8 -*-

import json
import pickle
import re

# from pytrie import SortedStringTrie as Trie
from pypinyin import pinyin, lazy_pinyin, Style


def creat_dict(word_li):
    # 该其中一个为-,全部正向建立词典
    res_dict = {}
    for word in word_li:
        num = len(word)
        for i in range(num):
            tokens = list(word)
            tokens[i] = "_"
            key = "".join(tokens)
            if key not in res_dict:
                res_dict[key] = {word}
            else:
                res_dict[key].add(word)
    return res_dict


def get_cy_data():
    """ 批量写入数据 """
    fliter_set = {"再接再历", "号啕大哭", "按纳不住", "水楔不通"}
    save_set = set()
    path = "/data_local/TwoWaysToImproveCSC/BERT/chinese-xinhua/data/idiom.json"
    with open(path, 'r', encoding="utf-8") as f:
        chengyu_dict = json.load(f)
        for item in chengyu_dict:
            if item['word'] in fliter_set:
                continue
            save_set.add(item['word'])

    path_oulu = "/data_local/TwoWaysToImproveCSC/BERT/chinese-xinhua/idom_detail.pkl"
    chengyu_oulu = pickle.load(open(path_oulu, "rb"))
    for item in chengyu_oulu:
        if item["成语"] in fliter_set:
            continue
        else:
            save_set.add(item['成语'])
    print(len(save_set))  # 49548
    pickle.dump(save_set, open("cy_set.pkl", "wb"), protocol=0)
    return save_set


def get_ci_data_spacy():
    """
    用于词语是否有错检测
    :return:
    """
    spacy_save_set = set()
    zh_ci_dict = pickle.load(open("./vocab_xiaoxue.pkl", "rb"))
    for w in zh_ci_dict:
        spacy_save_set.add(w)
    print(len(spacy_save_set))  # 385830
    return spacy_save_set


def get_ci_data():
    """
    用于获取候选词语
    # 如果和前面用同一个会不会更好呢
    # 如果只用前面那个呢
    :return:
    汉语词典第七版
    """
    # vocab_dict = pickle.load(open("./vocab_xiaoxue.pkl", "rb"))
    # word_set = set()
    # for s in vocab_dict:
    #     if 2 <= len(s) < 4 and not re.search('[^\u4e00-\u9fa5]', s):
    #         word_set.add(s)
    # print(len(word_set))  # 20831

    # # 新华字典，里面有很多词没有，学生，父母，今天
    # fliter_set = {"再接再历", "号啕大哭", "按纳不住", "水楔不通"}
    # save_set = set()
    # path = "/data_local/TwoWaysToImproveCSC/BERT/chinese-xinhua/data/ci.json"
    # with open(path, 'r', encoding="utf-8") as f:
    #     chengyu_dict = json.load(f)
    #     for item in chengyu_dict:
    #         if item['ci'] in fliter_set:
    #             continue
    #         if 2 <= len(item['ci']) < 4 and not re.search('[^\u4e00-\u9fa5]', item['ci']):
    #             save_set.add(item['ci'])
    # print(len(save_set))  # 234240
    # pickle.dump(save_set, open("ci_set.pkl", "wb"), protocol=0)

    # XDHYCD7th.txt

    fliter_set = {}
    save_set = set()
    com_obj = re.compile("【(.*?)】")
    path = "/data_local/TwoWaysToImproveCSC/BERT/chinese-xinhua/data/XDHYCD7th.txt"
    with open(path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            tmp_li = com_obj.findall(line)
            for s in tmp_li:
                if s in fliter_set:
                    continue
                if 1 < len(s) < 4 and not re.search('[^\u4e00-\u9fa5]', s):
                    save_set.add(s)
    print(len(save_set))  # 50959
    return save_set


word_set = get_ci_data()
pickle.dump(word_set, open("ci_set.pkl", "wb"), protocol=0)
words_dict = creat_dict(word_set)
pickle.dump(words_dict, open("match_ci.pkl", "wb"), protocol=0)
print(words_dict["_净"])
print(words_dict["干_"])

spacy_word_set = get_ci_data_spacy()
pickle.dump(spacy_word_set, open("ci_set_spacy.pkl", "wb"), protocol=0)
spacy_words_dict = creat_dict(spacy_word_set)
pickle.dump(spacy_words_dict, open("match_ci_spacy.pkl", "wb"), protocol=0)
print(spacy_words_dict["_净"])
print(spacy_words_dict["干_"])

# cy_set = get_cy_data()
# cy_dict = creat_dict(cy_set)
# print(cy_dict["_以致用"])
# print(cy_dict["学_致用"])
# print(cy_dict["学以_用"])
# print(cy_dict["学以致_"])
# pickle.dump(cy_dict, open("match_cy.pkl", "wb"), protocol=0)

same_pinyin = {}
for word in spacy_word_set | word_set:
    # ori_sent_pinyin_li = lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)
    ori_sent_pinyin_li = lazy_pinyin(word)
    py_key = "_".join(ori_sent_pinyin_li)
    if py_key not in same_pinyin:
        same_pinyin[py_key] = set()
    same_pinyin[py_key].add(word)

print(same_pinyin["chong_zhi"])
print(same_pinyin["yan_jing"])
print(same_pinyin["zhi_dao"])
print(same_pinyin["zuo_chu"])
ori_sent_pinyin_li = lazy_pinyin("重置")
print(ori_sent_pinyin_li)

# -*- coding: UTF-8 -*-

import sys
import os
import re
from optparse import OptionParser
from tqdm import tqdm
import pickle
import numpy as np
import re
from strTools import uniform, is_chinese
from clean_baike_zh import ratio_alphabetic
import pickle

vob = set()
with open("/data_local/plm_models/chinese_L-12_H-768_A-12/vocab.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        vob.add(line.strip())


# def normalize_uniform(text):
#     """
#     # 非汉字全部半角化？
#     # 统一起来会比较好
#     :param text:
#     :return:
#     """
#     text = uniform(text).lower()
#     text = re.sub("\s+", "", text)
#
#     return text


def normalize_lower(text):
    """
    # 非汉字全部半角化？
    # 统一起来会比较好
    :param text:
    :return:
    """
    text = text.lower()
    text = re.sub("\s+", "", text)
    return text


# def normalize(text):
#     """
#     # 非汉字全部半角化？
#     # 统一起来会比较好
#     :param text:
#     :return:
#     """
#     text = re.sub("\s+", "", text)
#     return text


def readAllConfusionSet(filepath):
    with open(filepath, 'rb') as f:
        allSim = pickle.load(f)
        return allSim


class GenerateCSC:
    def __init__(self):
        self.confusion_set = readAllConfusionSet('/data_local/TwoWaysToImproveCSC/BERT/save/confusion.file')

        # # 完全符合作文的混淆集
        # with open('/data_local/TwoWaysToImproveCSC/BERT/save/cc_confuse.txt', "r") as f:
        #     text = f.read().strip().split("\n\n")
        #     for item_str in text:
        #         item_li = item_str.strip().split("\n")
        #         s, t = item_li[0].strip(), item_li[1].strip()
        #         print(s, t)
        #         if s in self.confusion_set:
        #             self.confusion_set[s].add(t)
        #         else:
        #             self.confusion_set[s] = set()
        #             self.confusion_set[s].add(t)

        print(type(self.confusion_set))
        vocab = pickle.load(open("/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_vocab.pkl", "rb"))
        self.vocab = [s for s in vocab if s in vob and is_chinese(s)]
        print(len(self.vocab))

    def replace_token(self, line):
        """
        # 随机选取id，查看id对应的词是否在候选集中
        # 如果在90%替换为候选词，10%随机选取token（来自哪里呢？统计字典）
        # 知道1/4的字符被替换
        :param line:
        :return:
        """
        num = len(line)
        tokens = list(line)
        index_li = [i for i in range(num)]
        # 有可能是引入的错误个数太多了
        up_num = 6
        # 有可能是引入的错误个数太多了
        np.random.shuffle(index_li)
        print(index_li)
        count = 0
        for i in index_li:
            if tokens[i] in self.confusion_set:
                count += 1
                print(tokens[i])
                if np.random.rand(1) > 0.9:  # 这个比例是否要控制
                    idx = np.random.randint(0, len(self.vocab))
                    tokens[i] = self.vocab[idx]
                else:
                    token_conf_set = self.confusion_set[tokens[i]]
                    idx = np.random.randint(0, len(token_conf_set))
                    tokens[i] = list(token_conf_set)[idx]
                print(tokens[i])
                if count == up_num:
                    break
            else:
                print(tokens[i])

        return "".join(tokens)


if __name__ == "__main__":
    obj = GenerateCSC()
    # 我觉得我们要相信自己的能力，别人可以做得到，我也可以，不要因为小小的挫折而放弃，像培根说的：「奇迹多是在厄运中出现的。」
    # 爱迪生，小时候被所有人骂他是低能儿，可是他永有不屈不挠向上的精神，从火海中再站起来，后来他成为了不起的发明家。
    with open("../BERT/data/wiki_00_base.train", "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = "爱迪生，小时候被所有人骂他是低能儿，可是他拥有不屈不挠向上的精神，从火海中再站起来，后来他成为了不起的发明家。。"
            print(line)
            res = obj.replace_token(line)
            print(res)
            break

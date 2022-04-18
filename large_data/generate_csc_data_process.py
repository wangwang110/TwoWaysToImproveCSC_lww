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
    def __init__(self, infile, outfile, vocab):
        self.infile = infile
        self.outfile = outfile
        self.corpus = []
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
        self.vocab = [s for s in vocab if s in vob and is_chinese(s)]
        print(len(self.vocab))
        self.read(self.infile)
        self.write(self.corpus, self.outfile)

    def read(self, path):
        print("reading now......")
        with open(path, encoding="UTF-8") as f:
            i = 0
            for line in f.readlines():
                i += 1
                line = normalize_lower(line)
                new_line = self.replace_token(line)
                self.corpus.append([line, new_line])
                if i % 10000 == 0:
                    print("++++{}++++".format(i / 85649593))

        print("read finished.")

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
        up_num = num // 4
        # 有可能是引入的错误个数太多了
        np.random.shuffle(index_li)
        count = 0
        for i in index_li:
            if tokens[i] in self.confusion_set:
                count += 1
                if np.random.rand(1) > 0.9:  # 这个比例是否要控制
                    idx = np.random.randint(0, len(self.vocab))
                    tokens[i] = self.vocab[idx]
                else:
                    token_conf_set = self.confusion_set[tokens[i]]
                    idx = np.random.randint(0, len(token_conf_set))
                    tokens[i] = list(token_conf_set)[idx]
                if count == up_num:
                    break

        return "".join(tokens)

    def write(self, list, path):
        print("writing now......")
        if os.path.exists(path):
            os.remove(path)
        file = open(path, encoding="UTF-8", mode="w")
        for item in list:
            line1, line2 = item
            file.writelines(line2.strip() + " " + line1.strip() + "\n")
        file.close()
        print("writing finished")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--path", dest="path", default="", help="path file")
    parser.add_option("--input", dest="input", default="", help="input file")
    parser.add_option("--output", dest="output", default="", help="output file")
    (options, args) = parser.parse_args()
    path = options.path
    input = options.input
    output = options.output
    vocab_dict = pickle.load(open(path, "rb"))

    try:
        GenerateCSC(infile=input, outfile=output, vocab=vocab_dict)
        print("All Finished.")
    except Exception as err:
        print(err)

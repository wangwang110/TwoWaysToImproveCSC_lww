# -*- coding: UTF-8 -*-

import sys
import os
import re
from optparse import OptionParser
from tqdm import tqdm
import pickle
import numpy as np
import re
from strTools import stringQ2B
import pickle


def normalize(text):
    """
    # 非汉字全部半角化？
    # 统一起来会比较好
    :param text:
    :return:
    """
    text = stringQ2B(text).lower()
    text = re.sub("\s+", "", text)

    table = {ord(f): ord(t) for f, t in zip(
        u'“”‘’—', u'\"\"\'\'-')}
    text = text.translate(table)
    text = text.replace("…", "=")

    return text


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
        # 增加自己整理的候选集
        path = "/data_local/TwoWaysToImproveCSC/BERT/易混淆词/condusion_collect.pkl"
        confusion_dict = pickle.load(open(path, "rb"))
        for key in confusion_dict:
            if key in self.confusion_set:
                self.confusion_set[key] = self.confusion_set[key] | confusion_dict[key]
            else:
                self.confusion_set[key] = confusion_dict[key]
                print(key)
                print(confusion_dict[key])

        # # 增加spellgcn的候选集
        # path = "/data_local/TwoWaysToImproveCSC/BERT/save/spellGraphs.txt"
        # with open(path, "r", encoding="utf-8") as f:
        #     for line in f.readlines():
        #         s, t, r = line.strip().split("|")
        #         if r not in ["同音同调", "同音异调", "形近"]:
        #             continue
        #         if s not in self.confusion_set:
        #             self.confusion_set[s] = set()
        #         self.confusion_set[s].add(t)
        #
        #         if t not in self.confusion_set:
        #             self.confusion_set[t] = set()
        #         self.confusion_set[t].add(s)

        print(type(self.confusion_set))
        self.vocab = [s for s in vocab]
        self.read(self.infile)
        self.write(self.corpus, self.outfile)

    def read(self, path):
        print("reading now......")
        # if os.path.isfile(path) is False:
        #     print("path is not a file")
        #     exit()
        with open(path, encoding="UTF-8") as f:
            for line in f.readlines():
                if 6 < len(line) < 160 and not line.startswith("）"):
                    new_line = self.replace_token(line)
                    self.corpus.append([line, new_line])

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
                if np.random.rand(1) > 0.9:
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
            #
            # line1 = normalize(line1)
            # line2 = normalize(line2)
            #
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
    # sum_val = sum([vocab_dict[key] for key in vocab_dict])
    # for key in vocab_dict:
    #     vocab_dict[key] = vocab_dict[key] / sum_val

    try:
        GenerateCSC(infile=input, outfile=output, vocab=vocab_dict)
        print("All Finished.")
    except Exception as err:
        print(err)

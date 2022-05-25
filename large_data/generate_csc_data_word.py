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
import jieba
from pypinyin import pinyin, lazy_pinyin, Style

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
    text = text.strip().lower()
    text = re.sub("\s+", "", text)
    text = re.sub("\ue40c", "", text)
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
        self.confusion_set_base = readAllConfusionSet(options.confpath)
        # 同音词
        self.same_pinyin_words = pickle.load(
            open("/data_local/TwoWaysToImproveCSC/large_data//data/same_pinyin_words.pkl", "rb"))

        # '/data_local/TwoWaysToImproveCSC/BERT/save/confusion.file'

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
        self.vocab = [s for s in vocab if s in vob and is_chinese(s)]

        self.confusion_set = {}
        for s in self.confusion_set_base:
            self.confusion_set[s] = set()
            for t in self.confusion_set_base[s]:
                if t in self.vocab:
                    self.confusion_set[s].add(t)
        print(type(self.confusion_set))

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
        up_num = int(num * options.ratio)
        # 有可能是引入的错误个数太多了
        np.random.shuffle(index_li)
        count = 0
        words = list(jieba.cut(line))
        for i in index_li:
            if np.random.rand(1) > 0.5 and i < num - 1 and tokens[i] + tokens[i + 1] in words:
                w_pinyin_li = lazy_pinyin(tokens[i] + tokens[i + 1])
                w_pinyin_str = "_".join(w_pinyin_li)
                if w_pinyin_str in self.same_pinyin_words:
                    word_conf_set = self.same_pinyin_words[w_pinyin_str]
                    if tokens[i] + tokens[i + 1] in word_conf_set:
                        word_conf_set.remove(tokens[i] + tokens[i + 1])
                    if len(word_conf_set) > 0:
                        idx = np.random.randint(0, len(word_conf_set))
                        word = list(word_conf_set)[idx]
                        if tokens[i] != word[0]:
                            tokens[i] = word[0]
                            count += 1
                        if tokens[i + 1] != word[1]:
                            tokens[i + 1] = word[1]
                            count += 1
            else:
                if tokens[i] in self.confusion_set and len(self.confusion_set[tokens[i]]) != 0:
                    if np.random.rand(1) > options.confuse_ratio:  # 这个比例是否要控制
                        idx = np.random.randint(0, len(self.vocab))
                        tokens[i] = self.vocab[idx]
                        count += 1
                    else:
                        token_conf_set = self.confusion_set[tokens[i]]
                        if len(token_conf_set) > 0:
                            idx = np.random.randint(0, len(token_conf_set))
                            tokens[i] = list(token_conf_set)[idx]
                            count += 1
            if count >= up_num:
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
    parser.add_option("--ratio", type=float, default=0.25, help="error ratio")
    parser.add_option("--confuse_ratio", type=float, default=0.9, help="error ratio")
    # parser.add_option("--new_confset", type=int, default=0, help="use newconfset")
    parser.add_option("--seed", type=int, default=10, help="random seed")
    parser.add_option("--confpath", dest="confpath", default="", help="confpath file")
    # parser.add_option("--vocabpath", dest="vocabpath", default="", help="vocabpath")

    (options, args) = parser.parse_args()
    path = options.path
    input = options.input
    output = options.output
    vocab_dict = pickle.load(open(path, "rb"))

    np.random.seed(options.seed)

    # try:
    GenerateCSC(infile=input, outfile=output, vocab=vocab_dict)
    print("All Finished.")
    # except Exception as err:
    #     print(err)

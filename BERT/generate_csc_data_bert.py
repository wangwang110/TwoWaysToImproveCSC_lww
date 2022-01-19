# -*- coding: UTF-8 -*-

import sys
import os
import re
from optparse import OptionParser
from tqdm import tqdm
import pickle
import numpy as np
from bert_eval import BertMlm, vob, b2v

bert_path = "/data_local/plm_models/chinese_L-12_H-768_A-12/"
obj = BertMlm(bert_path)


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

        print(type(self.confusion_set))
        self.vocab = [s for s in vocab]
        self.read(self.infile)
        self.write(self.corpus, self.outfile)

    def read(self, path):
        print("reading now......")
        all_texts = []
        with open(path, encoding="UTF-8") as f:
            for line in f.readlines():
                if 6 < len(line) < 160 and not line.startswith("）"):
                    all_texts.append(line.strip())
        # import time
        # t0 = time.time()
        model_output = obj.test_without_trg(all_texts)
        # print(model_output)

        for i in tqdm(range(len(all_texts))):
            line = all_texts[i]
            new_line = self.replace_token(line, model_output[i])
            self.corpus.append([line, new_line])

        print("read finished.")

    def replace_token(self, line, bert_ouput):
        """
        # 随机选取id，查看id对应的词是否在候选集中
        # 如果在90%替换为候选词，10%随机选取token（来自哪里呢？统计字典）
        # 知道1/4的字符被替换
        :param line:
        :return:
        """
        num = len(line)
        tokens = list(line)  # 使用bert的分词，才能一一对应起来
        index_li = [i for i in range(num)]
        up_num = num // 4
        np.random.shuffle(index_li)
        count = 0

        for i in index_li:
            if tokens[i] in self.confusion_set:
                count += 1
                if np.random.rand(1) > 0.9:
                    # 随机替换为词表中的词
                    idx = np.random.randint(0, len(self.vocab))
                    tokens[i] = self.vocab[idx]
                else:
                    # 替换为候选集中，最有可能出错的词，bert概率输出最大的，并且在候选集中的词

                    token_conf_set = list(self.confusion_set[tokens[i]])
                    id_li = [b2v[s] for s in token_conf_set]
                    id2prob = [bert_ouput[j] for j in id_li]
                    idx = np.argmax(id2prob)
                    tokens[i] = token_conf_set[idx]
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
            file.writelines(line1.strip() + " " + line2.strip() + "\n")
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

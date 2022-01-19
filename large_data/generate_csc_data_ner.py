# -*- coding: UTF-8 -*-

import sys
import os
import re
from optparse import OptionParser
from tqdm import tqdm
import pickle
import numpy as np
import hanlp


def readAllConfusionSet(filepath):
    with open(filepath, 'rb') as f:
        allSim = pickle.load(f)
        return allSim


class GenerateCSC:
    def __init__(self, infile, outfile, vocab):

        self.HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH,
                                tasks=['ner'])  # 世界最大中文语料库

        for task in self.HanLP.tasks.values():
            task.sampler_builder.batch_size = 256

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
        all_doc = self.HanLP(all_texts)
        # print(all_doc)
        # print(time.time() - t0)

        for i in tqdm(range(len(all_texts))):
            line = all_texts[i]
            words = all_doc["tok/fine"][i]
            ner_words = all_doc["ner/msra"][i]
            new_line = self.replace_token(line, words, ner_words)
            self.corpus.append([line, new_line])

        print("read finished.")

    def replace_token(self, line, words, ner_words):
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
        up_num = num // 4
        np.random.shuffle(index_li)
        count = 0

        ner_word_ids = []
        for item in ner_words:
            if item[1] == "DATE":
                continue
            start = sum([len(words[i]) for i in range(item[2])])
            ids = [j for j in range(start, start + len(item[0]))]
            ner_word_ids.extend(ids)

        for i in index_li:
            if i in ner_word_ids:
                # 不是实体，如果是实体，则跳过
                continue
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

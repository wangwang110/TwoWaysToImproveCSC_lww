# -*- coding: UTF-8 -*-

import sys
import os
import re
from optparse import OptionParser
from tqdm import tqdm
import pickle
import numpy as np
import hanlp

CUDA_VISIBLE_DEVICES = "7"


def readAllConfusionSet(filepath):
    with open(filepath, 'rb') as f:
        allSim = pickle.load(f)
        return allSim


class GenerateCSC:
    def __init__(self, infile, outfile):

        self.HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH,
                                tasks=['ner'])  # 世界最大中文语料库

        for task in self.HanLP.tasks.values():
            task.sampler_builder.batch_size = 256

        self.infile = infile
        self.outfile = outfile
        self.corpus = []
        self.read(self.infile)
        self.write(self.corpus, self.outfile)

    def read(self, path):
        print("reading now......")
        all_texts_src = []
        all_texts_trg = []
        with open(path, encoding="UTF-8") as f:
            for line in f.readlines():
                src, trg = line.strip().split(" ")
                all_texts_src.append(src)
                all_texts_trg.append(trg)
        # import time
        # t0 = time.time()
        all_doc = self.HanLP(all_texts_trg)
        # print(all_doc)
        # print(time.time() - t0)

        for i in tqdm(range(len(all_texts_trg))):
            src = all_texts_src[i]
            trg = all_texts_trg[i]
            words = all_doc["tok/fine"][i]
            ner_words = all_doc["ner/msra"][i]
            new_line = self.replace_token(trg, words, ner_words, src)
            self.corpus.append([new_line, trg])

        print("read finished.")

    def replace_token(self, line, words, ner_words, error_line):
        """
        # 随机选取id，查看id对应的词是否在候选集中
        # 如果在90%替换为候选词，10%随机选取token（来自哪里呢？统计字典）
        # 知道1/4的字符被替换
        :param line:
        :return:
        """
        num = len(line)
        ner_word_ids = []
        for item in ner_words:
            if item[1] == "DATE":
                continue
            start = sum([len(words[i]) for i in range(item[2])])
            ids = [j for j in range(start, start + len(item[0]))]
            ner_word_ids.extend(ids)
        src_tokens = list(error_line)
        trg_tokens = list(line)
        for i in range(num):
            if i in ner_word_ids and src_tokens[i] != trg_tokens[i]:
                src_tokens[i] = trg_tokens[i]
        return "".join(src_tokens)

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
    parser.add_option("--input", dest="input", default="", help="input file")
    parser.add_option("--output", dest="output", default="", help="output file")
    (options, args) = parser.parse_args()
    input = options.input
    output = options.output
    try:
        GenerateCSC(infile=input, outfile=output)
        print("All Finished.")
    except Exception as err:
        print(err)

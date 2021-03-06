# -*- coding: UTF-8 -*-

import sys
import os
import re
from optparse import OptionParser
from tqdm import tqdm
import pickle
import numpy as np
# import hanlp
import spacy


def readAllConfusionSet(filepath):
    with open(filepath, 'rb') as f:
        allSim = pickle.load(f)
        return allSim


class GenerateCSC:
    def __init__(self, infile, outfile, vocab):

        self.nlp = spacy.load("zh_core_web_trf",
                              disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

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
        # if os.path.isfile(path) is False:
        #     print("path is not a file")
        #     exit()
        all_texts = []
        with open(path, encoding="UTF-8") as f:
            for line in tqdm(f.readlines()[:100]):
                if 6 < len(line) < 160 and not line.startswith("）"):
                    all_texts.append(line.strip())

        all_doc = self.nlp.pipe(all_texts)
        import time
        t0 = time.time()
        for doc in tqdm(all_doc):
            print(doc)
            ent_li = []
            for ent in doc.ents:
                ent_li.append((ent.text, ent.start_char, ent.end_char, ent.label_))
            # text = [item.text for item in doc]
            line, new_line = self.replace_token(doc, ent_li)
            self.corpus.append([line, new_line])
        print(time.time() - t0)

        print("read finished.")

    def replace_token(self, doc, ent_li):
        """
        # 随机选取id，查看id对应的词是否在候选集中
        # 如果在90%替换为候选词，10%随机选取token（来自哪里呢？统计字典）
        # 知道1/4的字符被替换
        :param line:
        :return:
        """
        words = [item.text for item in doc]
        line = "".join(words)
        num = len(line)
        tokens = list(line)
        index_li = [i for i in range(num)]
        up_num = num // 4
        np.random.shuffle(index_li)
        count = 0

        # # a = self.HanLP([line])
        # words = tokens
        # ner_words = a["ner/msra"][0]
        # print(words)
        # print(ner_words)
        ner_word_ids = []

        for item in ent_li:
            if item[1] == "DATE":
                continue
            ids = [j for j in range(item[1], item[2])]
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

        return line, "".join(tokens)

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

    # try:
    GenerateCSC(infile=input, outfile=output, vocab=vocab_dict)
    print("All Finished.")
    # except Exception as err:
    #     print(err)

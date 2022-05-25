#
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pickle
import os


def preprocess_words(path_src_in, val=1):
    """
    :param path_in:
    :param path_out:
    :return:
    """
    text_dict = {}
    with open(path_src_in, "r") as f1:
        for line in f1:
            try:
                items = line.strip().split("\t")
                words = items[1:]
                num = len(words)
                if len(words[0]) != 2:
                    continue
                for i in range(num - 1):
                    if words[i] not in text_dict:
                        text_dict[words[i]] = set()
                    for j in range(i + 1, num):
                        if words[j] not in text_dict:
                            text_dict[words[j]] = set()
                        if val == 1:
                            if len(set(words[i]) & set(words[j])) == 1:
                                text_dict[words[i]].add(words[j])
                                text_dict[words[j]].add(words[i])
                        else:
                            if words[i] != words[j]:
                                text_dict[words[i]].add(words[j])
                                text_dict[words[j]].add(words[i])
            except Exception as e:
                print(line)
                continue

    return text_dict


path = "./chinese_homophone_word_simple.txt"
path_out = "./chinese_homophone_word_simple.pkl"
if os.path.exists(path_out):
    word_dict = pickle.load(open(path_out, "rb"))
    print(len(word_dict))
    print(word_dict["做出"])
    print(word_dict["多少"])
    print(word_dict["重置"])
    print(word_dict["眼镜"])
    print(word_dict["一副"])
    print(word_dict["回去"])
    print(word_dict["艺术"])
else:
    word_dict = preprocess_words(path)
    print(word_dict["做出"])
    print(word_dict["多少"])
    print(word_dict["充值"])
    print(word_dict["眼镜"])
    pickle.dump(word_dict, open(path_out, "wb"), protocol=0)

path = "./chinese_homophone_word_simple.txt"
path_out = "./chinese_homophone_word.pkl"
if os.path.exists(path_out):
    word_dict = pickle.load(open(path_out, "rb"))
    print(len(word_dict))
    print(word_dict["做出"])
    print(word_dict["多少"])
    print(word_dict["重置"])
    print(word_dict["眼镜"])
    print(word_dict["一副"])
    print(word_dict["回去"])
    print(word_dict["艺术"])
else:
    word_dict = preprocess_words(path, val=0)
    print(word_dict["做出"])
    print(word_dict["多少"])
    print(word_dict["充值"])
    print(word_dict["眼镜"])
    pickle.dump(word_dict, open(path_out, "wb"), protocol=0)

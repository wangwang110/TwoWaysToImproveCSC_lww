# coding: utf-8

"""
@File    : map2plome.py
@Time    : 2022/3/9 14:44
@Author  : liuwangwang
@Software: PyCharm
"""

import re
import json


def read_json(path_in, path_out):
    """

    :param path_in:
    :param path_out:
    :return:
    """
    with open(path_in, "r") as f, open(path_out, "w", encoding="utf-8") as fw:
        id = 0
        all_data = json.load(f)
        for item in all_data:
            try:
                src, trg = item["original_text"], item["correct_text"]
                fw.write(src + " " + trg + "\n")
            except:
                print(item)
            id += 1


def get_macbert_data(path_in, path_out):
    """

    :param path_in:
    :param path_out:
    :return:
    """
    with open(path_in, "r", encoding="utf-8") as f, open(path_out, "w", encoding="utf-8") as fw:
        id = 0
        all_data = []
        for line in f.readlines():
            try:
                src, trg = line.strip().split()
                tmp = {
                    "id": str(id),
                    "original_text": src,
                    "wrong_ids": [],
                    "correct_text": trg
                }
                i = 0
                for s, t in zip(list(src), list(trg)):
                    if s != t:
                        tmp["wrong_ids"].append(i)
                    i += 1
                all_data.append(tmp)
            except:
                print(line)
            id += 1

        json.dump(all_data, fw, ensure_ascii=False)


def get_plome_data(path_in, path_out):
    """

    :param path_in:
    :param path_out:
    :return:
    """
    with open(path_in, "r", encoding="utf-8") as f, open(path_out, "w", encoding="utf-8") as fw:
        for line in f.readlines():
            try:
                # print(line)
                src, trg = line.strip().split()
                new_src = " ".join(list(src))
                new_trg = " ".join(list(trg))
                fw.write(new_src + "\t" + new_trg + "\n")
            except:
                print(line)


def get_keep_data(path_in, path_out):
    """
    :param path_in:
    :param path_out:
    :return:
    """
    with open(path_in, "r", encoding="utf-8") as f, open(path_out, "w", encoding="utf-8") as fw:
        for line in f.readlines():
            try:
                src, trg = line.strip().split()
                new_src = list(src)
                new_trg = list(trg)
                res_trg = []
                for s, t in zip(new_src, new_trg):
                    if s == t:
                        res_trg.append("U")
                    else:
                        res_trg.append(t)

                fw.write("".join(new_src) + " " + "".join(res_trg) + "\n")
            except:
                print(line)


path_in = "./train.json"
path_out = "./train.txt"
read_json(path_in, path_out)

path_in = "./merge_train.txt"
path_out = "./merge_train.json"
get_macbert_data(path_in, path_out)

path_in = "./13test.txt"
path_out = "./13test.json"
get_macbert_data(path_in, path_out)

path_in = "./15test.txt"
path_out = "./15test.json"
get_macbert_data(path_in, path_out)

path_in = "./13train.txt"
path_out = "./13train.json"
get_macbert_data(path_in, path_out)

path_in = "./15train.txt"
path_out = "./15train.json"
get_macbert_data(path_in, path_out)

path_in = "./rep_autog_wang_train.txt"
path_out = "./rep_autog_wang_train.json"
get_macbert_data(path_in, path_out)

path_in = "./rep_autog_wang_1k_dev.txt"
path_out = "./rep_autog_wang_1k_dev.json"
get_macbert_data(path_in, path_out)

# path_in = "./13test_lower.txt"
# path_out = "./13test_lower_keep.txt"
# get_keep_data(path_in, path_out)

# path_in = "./13test_lower.txt"
# path_out = "./13test_lower_plome.txt"
#
# get_plome_data(path_in, path_out)
#
# path_in = "./13train_lower.txt"
# path_out = "./13train_lower_plome.txt"
#
# get_plome_data(path_in, path_out)
#
# path_in = "./15test.txt"
# path_out = "./15test_plome.txt"
#
# get_plome_data(path_in, path_out)
#
# path_in = "../cc_data/chinese_spell_lower_4.txt"
# path_out = "../cc_data/chinese_spell_lower_plome.txt"
#
# get_plome_data(path_in, path_out)

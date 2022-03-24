# coding: utf-8

"""
@File    : map2plome.py
@Time    : 2022/3/9 14:44
@Author  : liuwangwang
@Software: PyCharm
"""

import re


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


path_in = "./merge_train.txt"
path_out = "./merge_train_keep.txt"
get_keep_data(path_in, path_out)

path_in = "./13test_lower.txt"
path_out = "./13test_lower_keep.txt"
get_keep_data(path_in, path_out)

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

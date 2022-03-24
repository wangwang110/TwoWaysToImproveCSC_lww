# coding: utf-8

"""
@File    : map2plome.py
@Time    : 2022/3/9 14:44
@Author  : liuwangwang
@Software: PyCharm
"""

import re


#
# def test(path_in, path_out):
#     """
#     :param path_in:
#     :param path_out:
#     :return:
#     """
#     same = 0
#     not_same = 0
#     with open(path_in, "r", encoding="utf-8") as f, open(path_out, "w", encoding="utf-8") as fw:
#         for line in f.readlines():
#             try:
#                 if re.search('[\u4e00-\u9fa5]', line) is None:
#                     same += 1
#                 else:
#                     not_same += 1
#             except:
#                 print(line)
#         fw.write(path_in)
#         fw.write("有错：" + str(not_same))
#         fw.write("无错：" + str(same))
#         print(path_in)
#         print("有错：" + str(not_same))
#         print("无错：" + str(same))
#     return same, not_same


def get_dist_data(path_in, path_out):
    """
    :param path_in:
    :param path_out:
    :return:
    """
    same = 0
    not_same = 0
    with open(path_in, "r", encoding="utf-8") as f, open(path_out, "w", encoding="utf-8") as fw:
        for line in f.readlines():
            try:
                src, trg = line.strip().split()
                if src == trg:
                    same += 1
                else:
                    not_same += 1
            except:
                print(line)
        fw.write(path_in)
        fw.write("有错：" + str(not_same))
        fw.write("无错：" + str(same))
        print(path_in)
        print("有错：" + str(not_same))
        print("无错：" + str(same))
    return same, not_same


# path_in = "./TestTruth.txt"
# path_out = "./TestTruth_dist.txt"
# test(path_in, path_out)

path_in = "./15test.txt"
path_out = "./15test_dist.txt"

get_dist_data(path_in, path_out)

path_in = "./13test.txt"
path_out = "./13test_dist.txt"

get_dist_data(path_in, path_out)

path_in = "./13train.txt"
path_out = "./13train_dist.txt"

get_dist_data(path_in, path_out)

path_in = "../cc_data/chinese_spell_lower_4.txt"
path_out = "../cc_data/chinese_spell_lower_dist.txt"

get_dist_data(path_in, path_out)

path_in = "./13train_lower.txt"
path_out = "./13train_lower_plome.txt"

get_dist_data(path_in, path_out)

path_in = "./wiki_00_base.train"
path_out = "./wiki_00_base_dist.train"

get_dist_data(path_in, path_out)

path_in = "./merge_train.txt"
path_out = "./merge_train_dist.txt"

get_dist_data(path_in, path_out)

path_in = "./rep_autog_wang_train.txt"
path_out = "./rep_autog_wang_train_dist.txt"
get_dist_data(path_in, path_out)

"""
./13test.txt
有错：971
无错：29
../cc_data/chinese_spell_lower_4.txt
有错：497
无错：527
./13train_lower.txt
有错：339
无错：11
./new_pretrain_all.train
有错：9185935
无错：65

"""

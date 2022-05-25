# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import sys
import pickle

sys.path.append("..")
from model import readAllConfusionSet

"""
统计测试集错误分布情况

1. 音近,形近,其他,占比情况；在混淆集的占比情况

"""


def count(trains):
    """
    统计错误模式
    :param train:
    :return:
    """
    train_set = set()
    for train in trains:
        with open(train, "r", encoding="utf-8") as f:
            for line in f.readlines():
                try:
                    src, trg = line.strip().split()
                except Exception as e:
                    print(e)
                    continue
                if src == trg:
                    continue
                for s, t in zip(list(src), list(trg)):
                    if s != t:
                        train_set.add((s, t))
    return train_set


def get_error_ratio(test):
    """
    2. 句子个数，有错与无错占比；每句错误个数分布；句子长度
    """
    all_sents = 0
    error_sents = 0
    lengths = []
    error_num_li = []
    with open(test, "r", encoding="utf-8") as f:
        for line in f.readlines():
            try:
                src, trg = line.strip().split()
            except Exception as e:
                print(e)
                continue
            all_sents += 1
            lengths.append(len(src))
            num = 0
            if src == trg:
                continue
            error_sents += 1
            for s, t in zip(list(src), list(trg)):
                if s != t:
                    num += 1
            error_num_li.append(num)

    print("句子总数:{}, 错误句子占比：{}，正确句子占比:{}".format(all_sents, error_sents / all_sents,
                                                (all_sents - error_sents) / all_sents))
    print("句子平均长度:{}".format(sum(lengths) // all_sents))
    print("句子平均错误个数:{}".format(sum(error_num_li) / error_sents))


def get_confset_ratio(test_set, confusion_set_path):
    """
    1. 音近,形近,其他,占比情况；在混淆集的占比情况
    """
    print(confusion_set_path)

    # da_bert的混淆集
    confusion_set = pickle.load(open(confusion_set_path, "rb"))
    print(len(confusion_set))

    # spellgcn的混淆集
    path = "/data_local/TwoWaysToImproveCSC/BERT/save/spellGraphs.txt"
    pinyin_dict = {}
    gyp_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            s, t, r = line.strip().split("|")
            if r not in ["同音同调", "同音异调", "形近"]:
                continue
            if r in ["同音同调", "同音异调"]:
                if s not in pinyin_dict:
                    pinyin_dict[s] = set()
                pinyin_dict[s].add(t)

            if r in ["形近"]:
                if s not in gyp_dict:
                    gyp_dict[s] = set()
                gyp_dict[s].add(t)

    pinyin_num = 0
    xingzhuang_num = 0
    all_da_bert = 0

    for item in test_set:
        s, t = item[0], item[1]
        if t in pinyin_dict and s in pinyin_dict[t]:
            pinyin_num += 1
        elif t in gyp_dict and s in gyp_dict[t]:
            xingzhuang_num += 1

        if t in confusion_set and s in confusion_set[t]:
            all_da_bert += 1
            # 测试集中，在混淆集里面的错误模式
        else:
            continue
            # print(item)

    # 混淆集里面所有的错误模式
    all_da_bert_set = set()
    for s in confusion_set:
        for t in confusion_set[s]:
            all_da_bert_set.add((s, t))

    print("在dabert测试集中的占比:{}".format(round(all_da_bert / len(test_set), 3)))
    print("在dabert混淆集中的占比:{}".format(round(all_da_bert / len(all_da_bert_set), 3)))
    print("在spellgcn混淆集中的占比:{}".format((pinyin_num + xingzhuang_num) / len(test_set)))
    print("拼音错误占比:{}, 字形错误占比：{}，其他占比:{}".format(pinyin_num / len(test_set), xingzhuang_num / len(test_set),
                                                (len(test_set) - pinyin_num - xingzhuang_num) / len(test_set)))


zuowen_test = "/data_local/TwoWaysToImproveCSC/BERT/data/chinese_spell_4.pre"
zuowen_test_set = count([zuowen_test])
get_error_ratio(zuowen_test)
confset_path1 = '/data_local/TwoWaysToImproveCSC/BERT/save/confusion.file'
# confset_path2 = '/data_local/TwoWaysToImproveCSC/large_data/data/small_confuse_set_new.pkl'
# confset_path3 = '/data_local/TwoWaysToImproveCSC/large_data/data/large_confuse_set_new.pkl'
get_confset_ratio(zuowen_test_set, confset_path1)
# get_confset_ratio(zuowen_test_set, confset_path2)
# get_confset_ratio(zuowen_test_set, confset_path3)

# 占比只有0.6846652267818575
# 通过引入10%的随机词，占比可到0.8185745140388769
print("============================================")

sighan13_test = "/data_local/TwoWaysToImproveCSC/BERT/data/13test.pre"
sighan13_test_set = count([sighan13_test])
get_error_ratio(sighan13_test)
confset_path1 = '/data_local/TwoWaysToImproveCSC/BERT/save/confusion.file'
# confset_path2 = '/data_local/TwoWaysToImproveCSC/large_data/data/small_confuse_set_new.pkl'
# confset_path3 = '/data_local/TwoWaysToImproveCSC/large_data/data/large_confuse_set_new.pkl'
get_confset_ratio(sighan13_test_set, confset_path1)
# get_confset_ratio(sighan13_test_set, confset_path2)
# get_confset_ratio(sighan13_test_set, confset_path3)
# 0.9747340425531915
# 通过引入10%的随机词，占比可到0.9773936170212766

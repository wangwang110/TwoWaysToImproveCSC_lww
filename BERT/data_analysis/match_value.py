#
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


# def count_freq(trains):
#     """
#     统计错误模式
#     :param train:
#     :return:
#     """
#     train_set = {}
#     for train in trains:
#         with open(train, "r", encoding="utf-8") as f:
#             for line in f.readlines():
#                 try:
#                     src, trg = line.strip().split()
#                 except Exception as e:
#                     print(e)
#                     continue
#                 if src == trg:
#                     continue
#                 for s, t in zip(list(src), list(trg)):
#                     if s != t:
#                         key_str = s + "|" + t
#                         if key_str not in train_set:
#                             train_set[key_str] = 1
#                         else:
#                             train_set[key_str] += 1
#     return train_set


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


def calc_val(train, test):
    train_set = count(train)
    test_set = count(test)
    # print(test_set - train_set)

    return len(train_set & test_set) / len(test_set)


def get_ratio(trains, tests):
    test_set = count(tests)
    num = 0
    num_in = 0
    num_out = 0
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
                        num += 1
                        if (s, t) in test_set:
                            num_in += 1
                        else:
                            num_out += 1

    # print(num_in)
    # print(num_out)
    print(num_out / num_in)

    # train = ["/data_local/TwoWaysToImproveCSC/BERT/data/new_pretrain_all.train"]
    # test = ["/data_local/TwoWaysToImproveCSC/BERT/data/13test.txt"]
    # 1879005
    # 70293596
    # 37.41001008512484

    # 3624839
    # 142243558
    # 39.24134506387732

    return num_out / num_in


def confset_ratio_1():
    train = ["/data_local/TwoWaysToImproveCSC/BERT/data/wiki_00_base_0.train"]
    test = ["/data_local/TwoWaysToImproveCSC/BERT/data/chinese_spell_4.txt"]
    train_set = count(train)
    test_set = count(test)
    print(len(train_set & test_set) / len(test_set))
    get_ratio(train, test)

    train = ["/data_local/TwoWaysToImproveCSC/BERT/data/wiki_00_base_1.train"]
    train_set = count(train)
    test_set = count(test)
    print(len(train_set & test_set) / len(test_set))
    get_ratio(train, test)

    train = ["/data_local/TwoWaysToImproveCSC/BERT/data/wiki_00_base_2.train"]
    train_set = count(train)
    test_set = count(test)
    print(len(train_set & test_set) / len(test_set))

    get_ratio(train, test)

    # train = ["/data_local/TwoWaysToImproveCSC/BERT/data/new_pretrain_all.train"]
    # test = ["/data_local/TwoWaysToImproveCSC/BERT/data/13test.txt"]
    # 3528729
    # 735
    # 4800.991836734694

    # train = ["/data_local/TwoWaysToImproveCSC/BERT/data/new_pretrain.train"]
    # test = ["/data_local/TwoWaysToImproveCSC/BERT/data/13test.txt"]
    # 5449176
    # 735
    # 7413.844897959184

    # train = ["/data_local/TwoWaysToImproveCSC/BERT/data/new_pretrain_all_word.train"]
    # test = ["/data_local/TwoWaysToImproveCSC/BERT/data/13test.txt"]
    # 1628028
    # 740
    # 2200.0378378378377

    # train = ["/data_local/TwoWaysToImproveCSC/BERT/data/new_pretrain_all_word_1.train"]
    # test = ["/data_local/TwoWaysToImproveCSC/BERT/data/13test.txt"]
    # 2051450
    # 741
    # 2768.488529014845


def confset_ratio():
    train = [
        "/data_local/TwoWaysToImproveCSC/BERT/data/pretrain_auto.train",
        "/data_local/TwoWaysToImproveCSC/BERT/data/rep_autog_wang_train.txt",
        "/data_local/TwoWaysToImproveCSC/BERT/data/13train.txt"
    ]
    test = ["/data_local/TwoWaysToImproveCSC/BERT/data/13test.txt"]
    print(calc_val(train, test))


confset_ratio_1()

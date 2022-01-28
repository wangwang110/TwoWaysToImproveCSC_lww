# -*- coding: utf-8 -*-
import json
import re
import os


def remove_space(src_text):
    src_text = re.sub("\s+", "", src_text)
    return src_text


def find_ori_data(path):
    """
    获取原始数据，包括gold
    :param path:
    :return:
    """
    all_text_ori = []
    all_text_trg = []
    with open(path, encoding="utf-8") as f1:
        for line in f1.readlines():
            src, trg = line.strip().split()
            all_text_ori.append(src)
            all_text_trg.append(trg)
    return all_text_ori, all_text_trg


# # 原始数据
# data = 13
# path_13 = "/data_local/TwoWaysToImproveCSC/BERT/data/13test.txt"
# all_text_ori, all_text_trg = find_ori_data(path_13)

data = 4
path_4 = "/data_local/TwoWaysToImproveCSC/BERT/cc_data/chinese_spell_4.txt"
all_text_ori, all_text_trg = find_ori_data(path_4)

task1 = "self"
name1 = "wang2018_new"

task2 = "test"
name2 = "wang2018_punct"

model_out_path = "/data_local/TwoWaysToImproveCSC/BERT/data_analysis/" + task1 + "_" + str(name1) + "_" + str(
    data) + "_cor.txt"
all_text_paper_model = []
with open(model_out_path, encoding="utf-8") as f2:
    all_predicts = f2.readlines()
    for s in range(len(all_predicts)):
        line2 = all_predicts[s]
        key, tmp_text = line2.strip().split("[CLS]")
        trg_pre = tmp_text.strip("[SEP]").replace("[UNK]", "N").replace("UNK]", "N").replace("[UNK", "N")
        src_text = all_text_ori[s]

        trg_text = ""
        i = 1
        src_text = remove_space(src_text)
        num = len(src_text)
        for item in list(trg_pre.strip()):
            if item in ["N", "著", "："]:
                item = list(src_text)[i - 1]
            # elif list(src_text)[i - 1] in ["”", "“", '"', '"', "‘", "’"]:
            #     item = list(src_text)[i - 1]

            trg_text += item
            if i == num:
                break
            i += 1
        all_text_paper_model.append(trg_text.replace("##", ""))

model_out_path = "/data_local/TwoWaysToImproveCSC/BERT/data_analysis/" + task2 + "_" + str(name2) + "_" + str(
    data) + "_cor.txt"
all_text_model = []
with open(model_out_path, encoding="utf-8") as f2:
    all_predicts = f2.readlines()
    for s in range(len(all_predicts)):
        line2 = all_predicts[s]
        key, tmp_text = line2.strip().split("[CLS]")
        trg_pre = tmp_text.strip("[SEP]").replace("[UNK]", "N").replace("UNK]", "N").replace("[UNK", "N")
        src_text = all_text_ori[s]

        trg_text = ""
        i = 1
        src_text = remove_space(src_text)
        num = len(src_text)
        for item in list(trg_pre.strip()):
            if item in ["N", "著", "："]:
                item = list(src_text)[i - 1]
            # elif list(src_text)[i - 1] in ["”", "“", '"', '"', "‘", "’"]:
            #     item = list(src_text)[i - 1]
            trg_text += item
            if i == num:
                break
            i += 1
        context_string = trg_text.replace("##", "")

        table = {ord(f): ord(t) for f, t in zip(u'\"\"\'\'-',
                                                u'“”‘’—')}
        text = context_string.translate(table)
        text = text.replace("=", "…")

        all_text_model.append(text)

out_str = "A_" + task1 + "_" + name1 + task2 + "_" + name2
path_out = "/data_local/TwoWaysToImproveCSC/BERT/data_analysis/" + out_str + "_" + str(data) + ".txt"
with open(path_out, "w", encoding="utf-8") as fw3:
    for src, trg, pre, pre_ori in zip(all_text_ori, all_text_trg, all_text_model, all_text_paper_model):
        src = remove_space(src)
        trg = remove_space(trg)
        pre = remove_space(pre)
        pre_ori = remove_space(pre_ori)  # 论文放出的模型结果

        if pre == pre_ori:
            continue

        fw3.write(src + "\n")
        fw3.write(trg + "\n")
        fw3.write(pre_ori + "\n")
        fw3.write(pre + "\n")

        fw3.write("gold:\n")
        for j, x in enumerate(zip(src, trg)):
            s, t = x
            if s != t:
                fw3.write(str(j + 1) + "," + s + "," + t + "\n")

        fw3.write(task1 + "_" + name1 + ":\n")
        for j, x in enumerate(zip(src, pre_ori)):
            s, t = x
            if s != t:
                fw3.write(str(j + 1) + "," + s + "," + t + "\n")

        fw3.write(task2 + "_" + name2 + ":\n")
        for j, x in enumerate(zip(src, pre)):
            s, t = x
            if s != t:
                fw3.write(str(j + 1) + "," + s + "," + t + "\n")

        fw3.write("=================\n\n")

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


def get_model_output(model_out_path, all_text_ori):
    """
    :param path:
    :return:
    """
    all_model_outputs = []
    with open(model_out_path, encoding="utf-8") as f:
        all_predicts = f.readlines()
        for s in range(len(all_predicts)):
            src_text = all_text_ori[s]
            line = all_predicts[s]
            key, tmp_text = line.strip().split("[CLS]")
            trg_pre = tmp_text.strip("[SEP]").replace("[UNK]", "N")

            trg_text = ""
            i = 1
            src_text = remove_space(src_text)
            num = len(src_text)
            for item in list(trg_pre.strip()):
                if item in ["N"]:
                    item = list(src_text)[i - 1]
                trg_text += item
                if i == num:
                    break
                i += 1
            all_model_outputs.append(trg_text.replace("## ", ""))
    return all_model_outputs


# # 原始数据
# data = 13
# path_13 = "/data_local/TwoWaysToImproveCSC/BERT/data/13test.txt"
# all_text_ori, all_text_trg = find_ori_data(path_13)

data = 4
path_4 = "/data_local/TwoWaysToImproveCSC/BERT/cc_data/chinese_spell_cy_4.txt"
all_text_ori, all_text_trg = find_ori_data(path_4)

model_out_path = "/data_local/TwoWaysToImproveCSC/BERT/data_analysis/paper_preTrain_13test.txt_cor.txt"
all_model_outputs = get_model_output(model_out_path, all_text_ori)

model_out_path_1 = "/data_local/TwoWaysToImproveCSC/BERT/data_analysis/paper_preTrain_13test_cy.txt_cor.txt"
all_model_outputs_1 = get_model_output(model_out_path_1, all_text_ori)

task1 = "paper"
task2 = "paper"
name1 = "13test"
name2 = "13test_cy"
out_str = "A_" + task1 + "_" + name1 + "_" + task2 + "_" + name2

path_out = "/data_local/TwoWaysToImproveCSC/BERT/data_analysis/" + out_str + "_" + str(data) + ".txt"
with open(path_out, "w", encoding="utf-8") as fw3:
    for src, trg, pre, pre_ori in zip(all_text_ori, all_text_trg, all_model_outputs, all_model_outputs_1):
        src = remove_space(src)
        trg = remove_space(trg)
        pre = remove_space(pre)
        pre_ori = remove_space(pre_ori)  # 论文放出的模型结果

        if pre == pre_ori or pre == trg:
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

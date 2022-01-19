# -*- coding: utf-8 -*-
import json
import re
import os


def remove_space(src_text):
    src_text = re.sub("\s+", "", src_text)
    return src_text


# 原始数据
path = "/data_local/TwoWaysToImproveCSC/BERT/data/13test.txt"
all_text_ori = []
all_text_trg = []
with open(path, encoding="utf-8") as f1:
    for line in f1.readlines():
        src, trg = line.strip().split()
        all_text_ori.append(src)
        all_text_trg.append(trg)

# for name in ["baseline", "preTrain", "advTrain"]:
name = "test"
task = "ori"
model_out_path = "/data_local/TwoWaysToImproveCSC/BERT/data_analysis/" + task + "_" + str(name) + "_13_cor.txt"
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
            trg_text += item
            if i == num:
                break
            i += 1
        all_text_paper_model.append(trg_text)

# for name in ["baseline", "preTrain", "advTrain"]:
name = "test"
task = "seq"
model_out_path = "/data_local/TwoWaysToImproveCSC/BERT/data_analysis/" + task + "_" + str(name) + "_13_cor.txt"
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
            trg_text += item
            if i == num:
                break
            i += 1
        all_text_model.append(trg_text)

path_out = "/data_local/TwoWaysToImproveCSC/BERT/data_analysis/" + task + "_13.txt"
with open(path_out, "w", encoding="utf-8") as fw3:
    for src, trg, pre, pre_ori in zip(all_text_ori, all_text_trg, all_text_model, all_text_paper_model):
        src = remove_space(src)
        trg = remove_space(trg)
        pre = remove_space(pre)
        pre_ori = remove_space(pre_ori)
        # if trg == pre:
        #     continue
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

        fw3.write("ori:\n")
        for j, x in enumerate(zip(src, pre_ori)):
            s, t = x
            if s != t:
                fw3.write(str(j + 1) + "," + s + "," + t + "\n")

        fw3.write("seq:\n")
        for j, x in enumerate(zip(src, pre)):
            s, t = x
            if s != t:
                fw3.write(str(j + 1) + "," + s + "," + t + "\n")

        fw3.write("=================\n\n")

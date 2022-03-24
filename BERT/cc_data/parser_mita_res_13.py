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
            all_text_ori.append(remove_space(src))
            all_text_trg.append(remove_space(trg))
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
            trg_pre = tmp_text.replace("[SEP]", "").replace("##", "").replace("[UNK]", "N")

            trg_text = ""
            i = 1
            num = len(src_text)
            for item in list(trg_pre.strip()):
                if item in ["N"]:
                    item = list(src_text)[i - 1]
                trg_text += item
                if i == num:
                    break
                i += 1
            all_model_outputs.append(trg_text)
    return all_model_outputs


# # 原始数据

path = "/data_local/TwoWaysToImproveCSC/BERT/cc_data/chinese_spell_lower_4.txt"

# path = "/data_local/TwoWaysToImproveCSC/BERT/data/13test_lower.txt"
data = os.path.basename(path)
all_text_ori, all_text_trg = find_ori_data(path)

path_model = "/data_local/TwoWaysToImproveCSC/BERT/data/chinese_spell_lower_4_model.txt"
_, all_model_pre = find_ori_data(path_model)

path_mita = "/data_local/TwoWaysToImproveCSC/BERT/data/chinese_spell_lower_4_mita_punct.txt"
_, all_mita_pre = find_ori_data(path_mita)

path_out = "../data_analysis/compar_model2mita_1.txt"
with open(path_out, "w", encoding="utf-8") as fw3:
    for src, trg, pre, pre1 in zip(all_text_ori, all_text_trg, all_model_pre, all_mita_pre):
        src = remove_space(src)
        trg = remove_space(trg)
        pre = remove_space(pre)
        pre1 = remove_space(pre1)

        if pre == pre1 and trg == pre:
            # 模型和秘塔修改得一致
            continue

        fw3.write(src + "\n")
        fw3.write(trg + "\n")
        fw3.write(pre + "\n")
        fw3.write(pre1 + "\n")

        fw3.write("gold:\n")
        for j, x in enumerate(zip(src, trg)):
            s, t = x
            if s != t:
                fw3.write(str(j + 1) + " , " + s + " , " + t + "\n")

        fw3.write("model:\n")
        for j, x in enumerate(zip(src, pre)):
            s, t = x
            if s != t:
                fw3.write(str(j + 1) + " , " + s + " , " + t + "\n")

        fw3.write("mita:\n")
        for j, x in enumerate(zip(src, pre1)):
            s, t = x
            if s != t:
                fw3.write(str(j + 1) + " , " + s + " , " + t + "\n")

        fw3.write("=================\n\n")

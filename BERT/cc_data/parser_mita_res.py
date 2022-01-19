# -*- coding: utf-8 -*-
import json
import re
import os
import string
import random


def change2correct_mita(ori_sent, error_li):
    if len(error_li) == 0:
        return ori_sent
    else:
        sent_li = list(ori_sent)

        for item in error_li:
            if item["replaceText"] == "" or item["replaceText"] is None:
                continue
            ori = item["sourceText"]
            cor = item["replaceText"]
            start = item["start"]
            end = item["end"]

            if len(ori) != len(cor):
                continue
                # print(item)
                # print("暂时只考虑拼写")  # 英文的省略号占6个字符，中文占两个字符
            elif ori == ":" and cor == "：":
                continue
            else:
                cor_li = list(cor)
                ori_li = list(ori)

                # print(item)
                # print(ori_sent)
                # print([(i, item) for i, item in enumerate(sent_li)])

                if sent_li[start - 1] == ori_li[0]:
                    for p in range(start, start + len(ori)):
                        sent_li[p - 1] = cor_li[p - start - 1]
                elif sent_li[start] == ori_li[0]:
                    for p in range(start, start + len(ori)):
                        sent_li[p] = cor_li[p - start]
        return "".join(sent_li)


def remove_space(src_text):
    src_text = re.sub("\s+", "", src_text)
    return src_text


# 原始数据
path = "/data_local/TwoWaysToImproveCSC/BERT/cc_data/xiaoxue_tc_4_sent.txt"
with open(path, encoding="utf-8") as f1:
    # 全部的数据
    all_text_dict = {}
    for line in f1:
        try:
            # 有的数据里面包含多个空格,去掉不要
            # line = line.replace("\n", "")
            key, src_text = line.strip().split(" ")
            all_text_dict[key] = src_text
        except:
            continue

# 秘塔结果
all_mita_res = {}
path_dir = "/data_local/TwoWaysToImproveCSC/BERT/cc_data/"
for filename in os.listdir(path_dir):
    if filename.startswith("all_res_spell_mita_cc_") and filename.endswith(".json"):
        json_path = path_dir + filename
        with open(json_path, "r", encoding="utf-8") as f2:
            mita_res = json.load(f2)
            for key in mita_res:
                if key in all_text_dict:
                    ori_sent = all_text_dict[key]
                    if len(mita_res[key]) != 0:
                        mita_text = change2correct_mita(ori_sent, mita_res[key])
                        all_mita_res[key] = mita_text
                        # all_mita_res[key] = ori_sent
                    else:
                        all_mita_res[key] = ori_sent

# 秘塔结果
# all_gcn_res = {}
# with open("/data_local/csc/SpellGCN-master/CSC4_keep/xiaoxue_gcn.json","r",encoding="utf-8") as f:


# 模型结果，与原始文件对齐
# name = "pretrain"
# task = "self"

name = "preTrain"
task = "paper"

# for name in ["baseline", "preTrain", "advTrain"]:
model_out_path = "/data_local/TwoWaysToImproveCSC/BERT/data_analysis/" + task + "_" + str(name) + "_4_cor.txt"
all_model_res = {}
with open(model_out_path, encoding="utf-8") as f2:
    for line2 in f2.readlines():
        key, tmp_text = line2.strip().split("[CLS]")
        trg_pre = tmp_text.strip("[SEP]").replace("[UNK]", "N").replace("UNK]", "N").replace("[UNK", "N")
        if key in all_text_dict:
            src_text = all_text_dict[key]
        else:
            continue

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
        all_model_res[key] = trg_text

# 最终写入文件
path_out = "/data_local/TwoWaysToImproveCSC/BERT/data_analysis/" + task + "_" + str(name) + "_4.txt"
with open(path_out, "w", encoding="utf-8") as fw3:
    for key in all_mita_res:
        if key not in all_text_dict:
            continue
        src_text = all_text_dict[key]
        src_text = remove_space(src_text)
        mita_text = all_mita_res[key]
        mita_text = remove_space(mita_text)
        if key in all_model_res:
            model_text = all_model_res[key]
            model_text = remove_space(model_text)
            # if src_text == mita_text and src_text == model_text:
            #     # src_text = re.sub("四年级:\w*$", "", src_text)
            #     # src_text = re.sub("四年级：\w*$", "", src_text)
            #     #
            #     # mita_text = re.sub("四年级:\w*$", "", mita_text)
            #     # mita_text = re.sub("四年级：\w*$", "", mita_text)
            #     # if len(model_text) > 25 and len(mita_text) < 45:
            #     #     fw3.write(key + "\n")
            #     #     fw3.write(src_text + "\n")
            #     #     fw3.write(mita_text + "\n")
            #     #     fw3.write(str(-1) + "\n\n")
            #     continue

            if mita_text == model_text:
                # 秘塔和模型都改了，并且改得相同
                continue
                # fw3.write(key + "\n")
                # fw3.write(src_text + "\n")
                # fw3.write(mita_text + "\n")
                # for j, x in enumerate(zip(src_text, model_text)):
                #     s, t = x
                #     if s != t:
                #         fw3.write(str(j + 1) + "," + s + "," + t + "\n")
                #
                # fw3.write("=================\n\n")
            # elif src_text != mita_text:
            #     # 秘塔修改了，和模型不同
            #     continue
            # elif src_text != model_text:
            #     # 模型修改了，和秘塔不同
            #     if "「" in model_text:
            #         continue
            #
            #     tag = 0
            #     for j, x in enumerate(zip(src_text, model_text)):
            #         s, t = x
            #         if s != t:
            #             if '”' in s or '"' in s or '“' in s or "她" in s or "他" in s or \
            #                     "它" in s or "…" in s or "’" in s or "‘" in s or "#" in s \
            #                     or len(set(string.digits) & set(s)) != 0 or len(
            #                 set(string.ascii_uppercase) & set(s)) != 0 \
            #                     or len(set(string.ascii_lowercase) & set(s)) != 0:
            #                 tag = 1
            #                 break
            #     if tag == 1:
            #         continue
            #
            #     fw3.write(key + "\n")
            #     fw3.write(src_text + "\n")
            #     fw3.write(mita_text + "\n")
            #     fw3.write(model_text + "\n")
            #
            #     fw3.write("mital:\n")
            #     for j, x in enumerate(zip(src_text, mita_text)):
            #         s, t = x
            #         if s != t:
            #             fw3.write(str(j + 1) + "," + s + "," + t + "\n")
            #
            #     fw3.write("model:\n")
            #     for j, x in enumerate(zip(src_text, model_text)):
            #         s, t = x
            #         if s != t:
            #             fw3.write(str(j + 1) + "," + s + "," + t + "\n")
            #
            #     fw3.write("=================\n\n")
            else:
                fw3.write(key + "\n")
                fw3.write(src_text + "\n")
                fw3.write(mita_text + "\n")
                fw3.write(model_text + "\n")

                fw3.write("mital:\n")
                for j, x in enumerate(zip(src_text, mita_text)):
                    s, t = x
                    if s != t:
                        fw3.write(str(j + 1) + "," + s + "," + t + "\n")

                fw3.write("model:\n")
                for j, x in enumerate(zip(src_text, model_text)):
                    s, t = x
                    if s != t:
                        fw3.write(str(j + 1) + "," + s + "," + t + "\n")

                fw3.write("=================\n\n")

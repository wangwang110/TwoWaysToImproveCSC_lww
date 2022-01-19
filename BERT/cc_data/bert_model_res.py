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
path_out = "/data_local/TwoWaysToImproveCSC/BERT/cc_data/zuowen.test"
with open(path_out, "w", encoding="utf-8") as f3:
    for filename in os.listdir(path_dir):
        if filename.startswith("all_res_spell_mita_cc_") and filename.endswith(".json"):
            json_path = path_dir + filename
            with open(json_path, "r", encoding="utf-8") as f2:
                mita_res = json.load(f2)
                for key in mita_res:
                    if key in all_text_dict:
                        ori_sent = all_text_dict[key]
                        f3.write(key + " " + ori_sent + "\n")

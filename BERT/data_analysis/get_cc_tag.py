# Copyright 2018 CVTE . All Rights Reserved.
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import string
import re


def get_sent(f):
    all_text = []
    tmp_text = []
    for line in f.readlines():
        if line.strip() == "" and len(tmp_text) != 0:
            all_text.append(tmp_text)
            tmp_text = []
        elif line.strip() != "":
            tmp_text.append(line.strip())

    return all_text


cc_tag_res = {}

with open("./chinese_spell.txt", "r", encoding="utf-8") as f:
    all_text = get_sent(f)

    error_all_texts = []
    for item in all_text:
        if item[-1].strip() != "-1":
            error_all_texts.append(item)
        key = item[0].strip()
        if key not in cc_tag_res:
            cc_tag_res[key] = item
        else:
            print(key)

# 加上来自真实作文的图片数据
with open("./zuowen_prallell.txt", "r", encoding="utf-8") as f:
    all_text_pic = get_sent(f)
    num = len(all_text_pic)
    for i in range(num):
        src, trg = all_text_pic[i]
        if src == trg:
            continue
        if len(set(src) & set(string.ascii_letters)) != 0:
            # 过滤掉拼音
            continue

        key = "(xiaoxue_tc_p_" + str(len(cc_tag_res)) + ")"
        cc_tag_res[key] = [key, src, trg]
        num_error = 0
        idx = 1
        for s, t in zip(src, trg):
            if s != t:
                num_error += 1
                cc_tag_res[key].append(str(idx) + "," + s + "," + t)
            idx += 1
        if num_error == 0:
            cc_tag_res[key].append("-1")
        else:
            error_all_texts.append(src)

with open("./chinese_spell_tag.txt", "w", encoding="utf-8") as fw:
    sorted_cc_tag_res = sorted(cc_tag_res.values(), key=lambda s: int(s[0].strip(")").strip("(").split("_")[-1]),
                               reverse=False)
    for item in sorted_cc_tag_res:
        for s in item:
            fw.write(s + "\n")
        fw.write("\n")

# 平行语料
with open("./chinese_spell_4.txt", "w", encoding="utf-8") as fw:
    for key in cc_tag_res:
        src = cc_tag_res[key][1]
        trg = cc_tag_res[key][2]
        error_li = []
        for s in cc_tag_res[key][3:]:
            error_li.append(s)

        if error_li[0] == "-1":
            if src == trg:
                fw.write(src + " " + trg + "\n")
            else:
                print("=====")
                print(src)
                print(trg)
        else:
            tokens = list(src)
            for item in error_li:
                pos, e, t = item.strip().split(",")
                tokens[int(pos) - 1] = t
            if "".join(tokens) == trg:
                fw.write(src + " " + trg + "\n")
            else:
                print(src)
                print(trg)

print(len(error_all_texts))

print(len(all_text) - len(error_all_texts))

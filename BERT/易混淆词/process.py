# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import re
from collections import defaultdict
from pypinyin import pinyin, lazy_pinyin, Style

map_dict = defaultdict(set)

#####################
path = "./1.txt"
with open(path, "r", encoding="utf-8") as f:
    for line in f.readlines():
        if re.match("\d+\.", line) is not None:
            line = re.sub("\d+\.", "", line.strip())
            item_li = line.strip().split(" ")
            num = len(item_li)
            for i in range(num - 1):
                for j in range(i + 1, num):
                    src = item_li[i]
                    trg = item_li[j]
                    for s, t in zip(list(src), list(trg)):
                        if s != t:
                            map_dict[s].add(t)
                            map_dict[t].add(s)
######################
# 需要拼音和字形辅助，字形人工过
path = "./2.txt"
with open(path, "r", encoding="utf-8") as f:
    for line in f.readlines():
        if re.match("\d{3}．", line) is not None:
            for text in line.strip().split(" "):
                text = re.sub("\d{3}．", " ", text.strip())
                for item in text.strip().split(" "):
                    words, error_str = re.split("\(|（", item)
                    t_li = error_str.strip("）").strip(")")
                    ch_pinyin_s = lazy_pinyin(words)
                    for t in t_li:
                        id = 0
                        tag = 0
                        ch_pinyin_t = lazy_pinyin(t)
                        for sitem in ch_pinyin_s:
                            if sitem in ch_pinyin_t:
                                tag = 1
                                map_dict[t].add(words[id])
                                map_dict[words[id]].add(t)
                            id += 1
                        if tag == 0:
                            print(item)
######################
path = "./3.txt"
with open(path, "r", encoding="utf-8") as f:
    for line in f.readlines():
        if "——" in line:
            item_li = list(re.sub("——", "", line.strip()))
            num = len(item_li)
            for i in range(0, num, 2):
                map_dict[item_li[i]].add(item_li[i + 1])
                map_dict[item_li[i + 1]].add(item_li[i])

######################
path = "./4.txt"
with open(path, "r", encoding="utf-8") as f:
    lines = []
    for line in f.readlines():
        if line.strip() == "" or "、" in line:
            continue
        lines.append(line)
    num = len(lines)
    for i in range(0, num, 2):
        src = re.sub("\(.*?\)", "", lines[i])
        trg = re.sub("\(.*?\)", "", lines[i + 1])
        if len(src) != len(trg):
            continue
        for s, t in zip(list(src), list(trg)):
            if s != t:
                map_dict[s].add(t)
                map_dict[t].add(s)

######################
path = "./5.txt"
with open(path, "r", encoding="utf-8") as f:
    lines = []
    for line in f.readlines():
        line = line.strip().split("：“")[0]
        try:
            if "/" in line:
                line = re.sub("\d+、", "", line.strip()).strip()
                src, trg = line.split("/")
                for s, t in zip(list(src), list(trg)):
                    if s != t:
                        map_dict[s].add(t)
                        map_dict[t].add(s)
        except Exception as e:
            print(line)

######################
path = "./6.txt"
with open(path, "r", encoding="utf-8") as f:
    lines = []
    for line in f.readlines():
        line = line.strip().split("：“")[0]
        item_li = re.split("\d+．", line)
        for item in item_li:
            if item == "":
                continue
            item = re.sub("\s+", "", item)
            ss, tt = re.split("（|\(", item)
            s = ss[-1]
            t = tt[0]
            if s != t:
                map_dict[s].add(t)
                map_dict[t].add(s)
######################
path = "./7.txt"
with open(path, "r", encoding="utf-8") as f:
    lines = []
    for line in f.readlines():
        if line.strip() == "":
            continue
        text = re.findall("【(.*?)】", line)
        src, trg = text[0].split("与")
        for s, t in zip(list(src), list(trg)):
            if s != t:
                map_dict[s].add(t)
                map_dict[t].add(s)

######################
path = "./8.txt"
with open(path, "r", encoding="utf-8") as f:
    lines = []
    for line in f.readlines():
        if line.strip() == "":
            continue
        text = re.findall("【(.*?)】", line)
        src, trg = text[0].split("与")
        for s, t in zip(list(src), list(trg)):
            if s != t:
                map_dict[s].add(t)
                map_dict[t].add(s)

import pickle

pickle.dump(map_dict, open("condusion_collect.pkl", "wb"), protocol=0)
print(map_dict)

with open("condusion_collect.txt", "w", encoding='utf-8') as f:
    for key in map_dict:
        f.write(key + "\n")
        f.write(" ".join(map_dict[key]) + "\n")
        f.write("\n")

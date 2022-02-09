# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import string
import re
from collections import defaultdict
from pypinyin import pinyin, lazy_pinyin, Style

map_dict = defaultdict(set)
with open("condusion_collect.txt", "r", encoding='utf-8') as f:
    text = f.read()
    for item in text.strip().split("\n\n"):
        if item.strip() != "":
            sitem_li = item.strip().split("\n")
            if len(set(sitem_li[0]) & set(string.digits)) != 0:
                print(sitem_li[0])
                continue
            map_dict[sitem_li[0]] = map_dict[sitem_li[0]] | set(sitem_li[1].strip().split(" "))

import pickle

pickle.dump(map_dict, open("condusion_collect_new.pkl", "wb"), protocol=0)
print(len(map_dict))
print(map_dict)

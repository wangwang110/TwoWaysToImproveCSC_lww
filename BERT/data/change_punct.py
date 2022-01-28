# Copyright 2018 CVTE . All Rights Reserved.
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unicodedata
import sys
import re


def special_token(context_string):
    """
    将bert词表中没有的中文标点转成英文标点
    :param context_string:
    :return:
    """
    # table = {ord(f): ord(t) for f, t in zip(
    #     u'“”‘’′，、。．！？【】（）％＃＠＆１２３４５６７８９０',
    #     u'\"\"\'\'\',,..!?[]()%#@&1234567890')}

    table = {ord(f): ord(t) for f, t in zip(
        u'“”‘’—', u'\"\"\'\'-')}
    text = context_string.translate(table)
    text = text.replace("…", "=")
    return text


# t = u'中国，中文，标点符号！你好？１２３４５＠＃【】+=-（）'
# t = "在父亲节的这个节日里，我想对爸爸说一声，我从未对他说过的一句话就是：“我爱你!”在此也祝所有天下的父亲节日快乐。』"
# t2 = unicodedata.normalize('NFKC', t)
# print(t2)
# 在父亲节的这个节日里,我想对爸爸说一声,我从未对他说过的一句话就是:“我爱你!”在此也祝所有天下的父亲节日快乐。』

path = sys.argv[1]
path_out = sys.argv[2]
with open(path, "r", encoding="utf-8") as f, open(path_out, "w", encoding="utf-8") as fw:
    for line in f.readlines():
        try:
            src, trg = line.strip().split(" ")
            src = special_token(src)
            src = re.sub("\s+", "", src)
            trg = special_token(trg)
            trg = re.sub("\s+", "", trg)
            # fw.write(src + " " + trg + "\n")
            fw.write(unicodedata.normalize('NFKC', src) + " " + unicodedata.normalize('NFKC', trg) + "\n")
        except Exception as e:
            print(e)
            print(line)

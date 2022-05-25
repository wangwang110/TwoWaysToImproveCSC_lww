# coding: utf-8

import re
import json
from strTools import uniform


def ratio_alphabetic(context_string):
    """
    :param context_string:
    :return: 返回中文字符（不包括标点）占比
    """
    # 标点
    # '[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\u4e00-\u9fa5]'
    t = re.findall('[\u4e00-\u9fa5]', context_string)
    num = len(context_string)
    num_chinese = len(''.join(t))
    if num == 0:
        return 0
    ratio = num_chinese * 1.0 / num
    return ratio


def find_csc(path_in, path_src, path_trg):
    """
    :param path_in:
    :param path_out:
    :return:
    """
    with open(path_in, "r") as f, open(path_src, "w", encoding="utf-8") as fw_src, \
            open(path_trg, "w", encoding="utf-8") as fw_trg:
        for line in f:
            item = json.loads(line)
            # print(item)
            src, trg = uniform(item["source"]), uniform(item["target"])
            if len(src) == len(trg):
                fw_src.write(src + "\n")
                fw_trg.write(trg + "\n")


def preprocess_data(path_in, path_src, path_trg):
    """
    :param path_in:
    :param path_out:
    :return:
    """
    with open(path_in, "r") as f, open(path_src, "w", encoding="utf-8") as fw_src, \
            open(path_trg, "w", encoding="utf-8") as fw_trg:
        for line in f:
            item = json.loads(line)
            # print(item)
            src, trg = uniform(item["source"]), uniform(item["target"])
            if len(src) == len(trg):
                fw_src.write(src + "\n")
                fw_trg.write(trg + "\n")


def preprocess_hsk(path_src_in, path_trg_in, path_src, path_trg):
    """
    :param path_in:
    :param path_out:
    :return:
    """
    with open(path_src_in, "r") as f1, open(path_trg_in, "r") as f2, open(path_src, "w", encoding="utf-8") as fw_src, \
            open(path_trg, "w", encoding="utf-8") as fw_trg:
        srcs = []
        for line in f1:
            src = re.sub("\s+", "", uniform(line))
            srcs.append(src)

        trgs = []
        for line in f2:
            trg = re.sub("\s+", "", uniform(line))
            trgs.append(trg)

        for src, trg in zip(srcs, trgs):
            if len(src) == len(trg):
                fw_src.write(src + "\n")
                fw_trg.write(trg + "\n")


def preprocess_ctc_valid(path_src_in, path_trg_in, path_out):
    """
    :param path_in:
    :param path_out:
    :return:
    """
    with open(path_src_in, "r") as f1, open(path_trg_in, "r") as f2, open(path_out, "w", encoding="utf-8") as fw:
        text_dict = {}
        for line in f1:
            key, sent = line.strip().split("\t")
            text_dict[key] = sent

        for line in f2.readlines():
            item = line.strip().split(",")
            key = item[0]
            sent = text_dict[key]
            tokens = list(sent)
            edits = [s.strip() for s in item[1:-1]]
            num = len(edits)
            try:
                for i in range(0, num, 4):
                    if len(edits[i + 2]) == len(edits[i + 3]):
                        # 11, 别字, 我来, 我发,
                        pos = int(edits[i])
                        trg_str = edits[i + 3]
                        for j in range(pos, pos + len(trg_str)):
                            tokens[j] = trg_str[j - pos]

                if "".join(tokens) != sent:
                    fw.write(sent.lower() + " " + "".join(tokens).lower() + "\n")
            except Exception as e:
                print(line)
                continue


def preprocess_faspell(path_src_in, path_trg_in, path_out):
    """
    :param path_in:
    :param path_out:
    :return:
    """
    with open(path_src_in, "r") as f1, open(path_trg_in, "r") as f2, open(path_out, "w", encoding="utf-8") as fw:
        for line in f1.readlines():
            try:
                item = line.strip().split("\t")
                fw.write(item[1].lower() + " " + item[2].lower().lower() + "\n")
            except Exception as e:
                print(line)
                continue
        for line in f2.readlines():
            try:
                item = line.strip().split("\t")
                fw.write(item[1].lower() + " " + item[2].lower().lower() + "\n")
            except Exception as e:
                print(line)
                continue


# path_in = "./train_large_v2.json"
# path_src = "./train_large_v2.src"
# path_trg = "./train_large_v2.trg"
# preprocess_data(path_in, path_src, path_trg)
#
# path_src_in = "./zh-hsk/hsk.src"
# path_trg_in = "./zh-hsk/hsk.trg"
# path_src = "./zh-hsk/hsk_csc.src"
# path_trg = "./zh-hsk/hsk_csc.trg"
# preprocess_hsk(path_src_in, path_trg_in, path_src, path_trg)

path_src_in = "./ctc/qua_input.txt"
path_trg_in = "./ctc/qua.solution"
path_out = "./ctc/qua_csc.txt"
preprocess_ctc_valid(path_src_in, path_trg_in, path_out)

path_src_in = "./FASPell/ocr_test_1000.txt"
path_trg_in = "./FASPell/ocr_train_3575.txt"
path_out = "./FASPell/ocr_train.csc"
preprocess_faspell(path_src_in, path_trg_in, path_out)

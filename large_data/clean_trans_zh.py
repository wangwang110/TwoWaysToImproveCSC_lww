# -*- coding: UTF-8 -*-

import os
from optparse import OptionParser
import json
import re


def cut_sent(para):
    """规则分句"""
    para = re.sub("([。！？!\?]){2,}", r"\1", para)
    para = re.sub('([。！？!\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    #
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？!\?][”’])([^，。！？!\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


def read_json_trans(path):
    """
    读取翻译数据
    :param path:
    :return:
    """
    data = set()
    if path.endswith("zh"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data.add(line.strip())
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                item = json.loads(line)
                data.add(item["chinese"])
    return data


def ratio_alphabetic(context_string):
    """
    :param context_string:
    :return: 返回中文字符个数占比
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


class Clean(object):
    def __init__(self, infile, outfile):
        self.infile = infile
        self.outfile = outfile
        self.corpus = set()
        self.read(self.infile)
        self.write(self.corpus, self.outfile)

    def read(self, path):
        print("reading now......")
        data = read_json_trans(path)
        i = 0
        data_num = len(data)
        for line in data:
            i += 1
            line = line.strip()
            if line == "":
                continue
            sent = self.process_line(line)
            if 6 < len(sent) < 160 and ratio_alphabetic(sent) > 1 / 2:
                self.corpus.add(sent)
            if i % 10000 == 0:
                print("########{}######".format(i / data_num))
        print("read finished.")

    def process_line(self, line):
        line = re.sub("\s+", "", line)
        line = re.sub("<br>", "", line)
        line = re.sub("\(\)", "", line)
        line = re.sub("(^【.*?】)", "", line)
        return line.strip().lower()

    def write(self, list, path):
        print("writing now......")
        if os.path.exists(path):
            os.remove(path)
        file = open(path, encoding="UTF-8", mode="w")
        for line in list:
            file.writelines(line + "\n")
        file.close()
        print("writing finished")


if __name__ == "__main__":
    print("clean corpus")
    parser = OptionParser()
    parser.add_option("--input", dest="input", default="", help="input file")
    parser.add_option("--output", dest="output", default="", help="output file")
    (options, args) = parser.parse_args()
    input = options.input
    output = options.output
    try:
        Clean(infile=input, outfile=output)
        print("All Finished.")
    except Exception as err:
        print(err)

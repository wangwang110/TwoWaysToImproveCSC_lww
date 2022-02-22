# -*- coding: UTF-8 -*-

import sys
import os
import re
from optparse import OptionParser
from tqdm import tqdm
import json


def read_json_trans(path):
    data = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            data.add(item["chinese"])
    return data


class Clean(object):
    def __init__(self, infile, outfile):
        self.infile = infile
        self.outfile = outfile
        self.corpus = []
        self.read(self.infile)
        self.write(self.corpus, self.outfile)

    def read(self, path):
        print("reading now......")
        data = read_json_trans(path)
        for line in data:
            if line.strip() == "":
                continue
            # 必须包含中文汉字
            if not re.search('[\u4e00-\u9fa5]', line.strip()):
                continue
            if 6 < len(line) < 160:
                text = self.process_line(line)
                self.corpus.append(text)
        print("read finished.")

    def process_line(self, line):
        line = re.sub("\s+", "", line)
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

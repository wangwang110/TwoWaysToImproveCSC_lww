# -*- coding: UTF-8 -*-

import os
import re
import collections
import pickle

path = "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/"
path_vocab = "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_vocab.pkl"

final_dict = {}
for file in os.listdir(path):
    filename = path + file
    print(filename)
    f = open(filename)  # 返回一个文件对象

    line = f.readline()  # 调用文件的 readline()方法
    i = 0
    while line:
        try:
            line = re.sub("\s+", "", line).strip()
            a = collections.Counter(list(line))
            for key in a:
                if key in final_dict:
                    final_dict[key] += 1
                else:
                    final_dict[key] = 1
            i += 1
            line = f.readline()
        except UnicodeError:
            break

new_final_dict = {}
for key in final_dict:
    if (key >= u'\u4e00') and (key <= u'\u9fa5'):
        new_final_dict[key] = final_dict[key]
pickle.dump(new_final_dict, open(path_vocab, "wb"), protocol=0)
print(new_final_dict)
print(len(new_final_dict))

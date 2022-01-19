# -*- coding: UTF-8 -*-
import os
import re
import collections
import pickle
from tqdm import tqdm
from clean_wiki_zh import ratio_alphabetic

path = "/data_local/TwoWaysToImproveCSC/large_data/weibo-400w/"
path_out = "/data_local/TwoWaysToImproveCSC/large_data/weibo/weibo_clean_second.txt"
path_vocab = "/data_local/TwoWaysToImproveCSC/large_data/weibo/weibo_vocab.pkl"
# encoding='ISO-8859-1'


first_stage_data = set()
second_stage_data = set()
with open("/data_local/TwoWaysToImproveCSC/large_data/tmp.txt", "r", encoding="utf-8") as ff:
    for line in ff.readlines():
        first_stage_data.add(line.strip())

final_dict = {}
for file in os.listdir(path):
    filename = path + file
    print(filename)
    with open(filename, 'r', encoding='ISO-8859-1') as f:
        for i in tqdm(f):
            try:
                x = i.encode('ISO-8859-1').decode('utf-8')
            except UnicodeError as e:
                print(e)
                x = ''

            if x != '':
                line = re.sub("\s{1,}", "", x)
                a = collections.Counter(list(line))
                #
                for key in a:
                    if key in final_dict:
                        final_dict[key] += 1
                    else:
                        final_dict[key] = 1
                #
                if 5 < len(line) < 160 and not line.startswith("）") \
                        and ratio_alphabetic(line) > 1 / 3 and line not in first_stage_data:
                    second_stage_data.add(line)

with open(path_out, "w", encoding="utf-8")  as fw:
    for s in second_stage_data:
        fw.write(s + "\n")

new_final_dict = {}
for key in final_dict:
    if (key >= u'\u4e00') and (key <= u'\u9fa5'):
        new_final_dict[key] = final_dict[key]
pickle.dump(new_final_dict, open(path_vocab, "wb"), protocol=0)
print(new_final_dict)
print(len(new_final_dict))

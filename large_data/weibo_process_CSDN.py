# -*- coding: UTF-8 -*-

import os
import re
from optparse import OptionParser
import collections
import pickle
from itertools import islice
from tqdm import tqdm

parser = OptionParser()
parser.add_option("--input", dest="input", default="", help="input file")
parser.add_option("--output", dest="output", default="", help="output file")
(options, args) = parser.parse_args()
path = options.input
path_out = options.output

# encoding='ISO-8859-1'
# 其实你已经离真相很近了
# 其实你已经离成功很近了
fw = open("/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/weibo.txt", "w", encoding="utf-8")  # 返回一个文件对象

final_dict = {}
for file in os.listdir(path):
    filename = path + file
    print(filename)

    with open(filename, 'r', encoding='ISO-8859-1') as f:
        for i in tqdm(f):
            # 此处将数据打印出来的时候我们会发现数据中中文部分会如上图一样。
            # print(i)
            # 因此处可能还是会因为数据中的特出字符导致报错，所以添加一个try在这里
            # 假如该条数据出错你可以选择不要或者选择将该条数据记录都行，这个看个人了。
            try:
                # 在这里我们将读取出来的数据先用 ‘ISO-8859-1’格式给它编码，然后通过‘utf-8’给它解码
                x = i.encode('ISO-8859-1').decode('utf-8')
            except Exception as e:
                print(e)
                x = ''

            if x != '':
                line = re.sub("\s{1,}", "", x)
                a = collections.Counter(list(line))
                for key in a:
                    if key in final_dict:
                        final_dict[key] += 1
                    else:
                        final_dict[key] = 1
                for key in a:
                    if key in final_dict:
                        final_dict[key] += 1
                    else:
                        final_dict[key] = 1
                if 6 < len(line) < 160:
                    fw.write(line+"\n")
fw.close()

print(final_dict)
print(len(final_dict))

new_final_dict = {}
i = 0
for key in final_dict:
    if (key >= u'\u4e00') and (key <= u'\u9fa5'):
        i += 1
        new_final_dict[key] = final_dict[key]
    if i == 50000:
        break
pickle.dump(new_final_dict, open(path_out, "wb"), protocol=0)
print(new_final_dict)
print(len(new_final_dict))

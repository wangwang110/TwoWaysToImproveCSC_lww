# coding: utf-8

"""
@File    : pinyin_process_data.py
@Time    : 2022/3/10 16:26
@Author  : liuwangwang
@Software: PyCharm
"""
# 统计声母，韵母，音调，拼音的个数
import pickle

from get_pinyin_data import sent2pinyin

path_in = "/data_local/TwoWaysToImproveCSC/BERT/data/new_pretrain.train"
# path_out = "/data_local/TwoWaysToImproveCSC/BERT/data/new_pretrain_pinyin.train"
pinyin_set = set()
inits_set = set()
finals_set = set()
tone_set = set()

num = 19460000
with open(path_in, "r", encoding="utf-8") as f1:
    i = 0
    for line in f1.readlines():
        i += 1
        src, trg = line.strip().split(" ")
        src_pinyin, src_inits, src_finals, src_tones = sent2pinyin(src)
        trg_pinyin, trg_inits, trg_finals, trg_tones = sent2pinyin(trg)
        #
        pinyin_set = pinyin_set | set(src_pinyin)
        pinyin_set = pinyin_set | set(trg_pinyin)

        inits_set = inits_set | set(src_inits)
        inits_set = inits_set | set(trg_inits)

        finals_set = finals_set | set(src_finals)
        finals_set = finals_set | set(trg_finals)

        tone_set = tone_set | set(src_tones)
        tone_set = tone_set | set(trg_tones)
        if i % 10000 == 0:
            print("=={}+++".format(i / num * 100))

# 输入是声母，韵母，音调
# 输出是拼音
pickle.dump(pinyin_set, open("./pinyin_set.bin", "wb"), protocol=0)
pickle.dump(inits_set, open("./inits_set.bin", "wb"), protocol=0)
pickle.dump(finals_set, open("./finals_set.bin", "wb"), protocol=0)
pickle.dump(tone_set, open("./tone_set.bin", "wb"), protocol=0)

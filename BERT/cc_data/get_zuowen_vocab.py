# coding: utf-8

"""
@File    : get_zuowen_vocab.py.py
@Time    : 2022/5/23 15:19
@Author  : liuwangwang
@Software: PyCharm
"""
import re
import pickle

vocab = {}
with open("./xiaoxue_sent_all.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        src = line.strip().split()[-1]
        for s in src:
            if not re.search('[\u4e00-\u9fa5]', s):
                continue
            if s not in vocab:
                vocab[s] = 1
            else:
                vocab[s] += 1
vocab_set = set()
for t in vocab:
    if vocab[t] > 1:
        vocab_set.add(t)

print(sorted(vocab.items(), key=lambda s: s[1], reverse=True))
print(len(vocab))
print(len(vocab_set))

pickle.dump(vocab_set, open("./zuowen_vocab.pkl", "wb"), protocol=0)

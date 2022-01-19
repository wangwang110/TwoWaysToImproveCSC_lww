# Copyright 2018 CVTE . All Rights Reserved.
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# import spacy
#
# nlp = spacy.load("zh_core_web_trf", disable=["parser"])
# # pipeline = ["tok2vec", "tagger", "parser", "ner"]
# sent_li = ["MIXION（迷炫作用），台湾男子音乐组合，由张翔然和雷小胤组成，两位成员均为混血儿。"]
#
# for sent in sent_li:
#     print(sent)
#     doc = nlp(sent)
#     for ent in doc.ents:
#         print(ent.text, ent.start_char, ent.end_char, ent.label_)
#     print("\n")


import hanlp

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH,
                   tasks=['ner'])  # 世界最大中文语料库

sent_li = ["MIXION（迷炫作用），台湾男子音乐组合，由张翔然和雷小胤组成，两位成员均为混血儿。"]
for line in sent_li:
    sent = line.strip().split(" ")[0]
    print(sent)
    a = HanLP([sent])
    print(a["ner/msra"])

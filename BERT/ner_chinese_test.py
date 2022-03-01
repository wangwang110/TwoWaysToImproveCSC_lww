#
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# import spacy
#
# nlp = spacy.load("zh_core_web_trf")
# doc = nlp("MIXION（迷炫作用），台湾男子音乐组合，由张翔然和雷小胤组成，两位成员均为混血儿。")
#
# for ent in doc.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)


import hanlp

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)  # 世界最大中文语料库

with open("/data_local/TwoWaysToImproveCSC/BERT/data/pretrain_auto.dev") as f:
    for line in f.readlines():
        # sent = line.strip().split(" ")[1]
        sent = "好浓郁的苏联风，好喜欢！《前进，达瓦里希》alink "
        print(sent)
        a = HanLP([sent,sent,sent,sent])
        print(a)
        print("\n")

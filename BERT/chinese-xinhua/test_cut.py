#
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import spacy

nlp = spacy.load("zh_core_web_sm")
text = "那块看似索然无味的黑板，其实是教室里众所曙目的大明星，班上每一个人都面对著它，无时无刻都盯著它看。 那块看似索然无味的黑板，其实是教室里众所瞩目的大明星，班上每一个人都面对著它，无时无刻都盯著它看。"
text = "互助组应当帮助鳏寡孤读"
text = "互助组应当帮助鳏寡孤读"
text = "盛夏的夜晚月色溶融,迎面又扑来一阵阵花香."
for w in nlp(text):
    print(w, w.tag_)

"""
import jieba

seg_list = jieba.cut_for_search("想不到他监守自盗，偷天换日，真是胆大忘为。")
seg_list = jieba.cut_for_search(
    "每个人的人生道路多少都会受点风雨，而这风雨不是逞罚而是挑战。 每个人的人生道路多少都会受点风雨，而这风雨不是惩罚而是挑战。")
print("cut_for_search  Mode: " + "/ ".join(seg_list))  # 全模式
# Full Mode: 想不到/ 不到/ 他/ 监守/ 监守自盗/ ，/ 偷天换日/ ，/ 真是/ 胆大/ 忘/ 为/ 。
# Default  Mode: 想不到/ 他/ 监守自盗/ ，/ 偷天换日/ ，/ 真是/ 胆大/ 忘为/ 。
# cut_for_search  Mode: 不到/ 想不到/ 他/ 监守/ 监守自盗/ ，/ 偷天换日/ ，/ 真是/ 胆大/ 忘为/ 。

"""

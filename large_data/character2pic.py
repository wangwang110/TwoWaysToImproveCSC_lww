#
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pickle

import pygame

pygame.init()
# 获取3500个汉字
with open("all_3500_chars.txt", "r", encoding="utf-8") as f:
    chars = f.read().strip()

pinyin_char = pickle.load(open("./data/same_pinyin_chars.pkl", "rb"))
char_li = []
for s in pinyin_char:
    char_li.extend(pinyin_char[s])

# 通过pygame将汉字转化为黑白图片
for char in char_li:
    font = pygame.font.Font("./STSONG.TTF", 100)
    rtext = font.render(char, True, (0, 0, 0), (255, 255, 255))
    pygame.image.save(rtext, "./char_pic_more/{}.png".format(char))

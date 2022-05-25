# coding: utf-8

"""
@File    : get_pinyin_data.py.py
@Time    : 2022/3/10 14:49
@Author  : liuwangwang
@Software: PyCharm
"""

import re
from pypinyin import pinyin, lazy_pinyin, Style
from pypinyin.contrib.tone_convert import to_initials, to_finals


def is_chinese(usen):
    """判断一个unicode是否是汉字"""
    for uchar in usen:
        if uchar >= '\u4e00' and uchar <= '\u9fa5':
            continue
        else:
            return False
    else:
        return True


class PinYin:
    def __init__(self):
        # 34
        # 去掉了 ün
        self.yun = ['a', 'o', 'e', 'i', 'u', 'er', 'ai', 'ei', 'ao', 'ou', 'ia', 'ie', 'ua', 'uo', 'iao', 'iou', 'iu',
                    'uai', 'ui', 'ue', 'an', 'en', 'in', 'ian', 'uan', 'un', 'ang', 'eng', 'ing', 'ong', 'iang', 'uang',
                    'ueng', 'iong']
        # 23
        sheng = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'z', 'c', 's', 'r', 'zh', 'ch',
                 'sh', 'y', 'w']
        self.sort_initials = sorted(sheng, key=lambda s: len(s), reverse=True)

        self.set_finals = set()

    def sent2pinyin(self, sent):
        """
        将句子转化为拼音，用句子转对多音字比较友好
        统计声母，韵母，音调
        :param sent:
        :return:
        """
        sent = re.sub("\s+", "", sent)
        tokens = list(sent)
        ori_sent_pinyin_li = lazy_pinyin(sent, style=Style.TONE3, neutral_tone_with_five=True)
        sent_initials = []
        sent_finals = []
        sent_tones = []

        if len(ori_sent_pinyin_li) != len(tokens):
            sent_pinyin_li = []
            num = len(ori_sent_pinyin_li)
            i = 0
            j = i
            while i < num and j < len(tokens):
                if is_chinese(tokens[j]) or lazy_pinyin(tokens[j], style=Style.TONE3, neutral_tone_with_five=True)[0] \
                        == ori_sent_pinyin_li[i]:
                    # 𣗋 is_chinese 返回False
                    sent_pinyin_li.append(ori_sent_pinyin_li[i])
                    i += 1
                    j += 1
                else:
                    for s in list(ori_sent_pinyin_li[i]):
                        sent_pinyin_li.append(s)
                    j = j + len(ori_sent_pinyin_li[i])
                    i = i + 1
        else:
            sent_pinyin_li = ori_sent_pinyin_li

        if len(sent_pinyin_li) != len(tokens):
            # print(sent)
            # print(ori_sent_pinyin_li)
            print("==========error===========")

        for s, w in zip(sent_pinyin_li, tokens):
            sent_tones.append(s[-1])
            # 通过匹配的方式自己获取声母，韵母
            if len(s) == 1:
                sent_initials.append(s)
                sent_finals.append(s)
            else:
                for c in self.sort_initials:
                    if s.startswith(c):
                        sent_initials.append(c)
                        sent_finals.append(s[len(c):-1])
                        break
        return sent_pinyin_li, sent_initials, sent_finals, sent_tones


# def sent2pinyin_init(sent):
#     """
#     将句子转化为拼音，用句子转对多音字比较友好（会将连续的非中文字符切成一个词）
#     统计声母，韵母，音调
#     :param sent:
#     :return:
#     """
#     sent = re.sub("\s+", "", sent)
#     tokens = list(sent)
#     sent_pinyin_li = lazy_pinyin(sent, style=Style.TONE3, neutral_tone_with_five=True)
#     if len(sent_pinyin_li) != len(tokens):
#         new_sent_pinyin_li = []
#         for s in tokens:
#             if
#
#     sent_initials = []
#     sent_finals = []
#     sent_tones = []
#
#     for s, w in zip(sent_pinyin_li, tokens):
#         start = to_initials(s)
#         if start != "":
#             sent_initials.append(start)
#         else:
#             sent_initials.append(w)
#
#         #
#         end = to_finals(s)
#         if end != "":
#             sent_finals.append(end)
#         else:
#             sent_finals.append(w)
#         sent_tones.append(s[-1])
#
#     return sent_pinyin_li, sent_initials, sent_finals, sent_tones
#

if __name__ == "__main__":
    obj = PinYin()

    a = obj.sent2pinyin(
        '在我们班柜上面，有许多第一名的奖杯，那些都是我们全体班上的恶新沥血，才艺竞赛、体育竞赛，都是第一名，让我回忆到大家合作的画面：「哔！开始！传球！这里！有空𣗋！投！进了！(场边传来一阵欢呼声)」。')
    print(a)
    # 无论那个都有一个unk和pad
    print(lazy_pinyin("你需要重置参数", style=Style.TONE3, neutral_tone_with_five=True))

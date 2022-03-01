# -*- coding: UTF-8 -*-

import re


def replace_space(para):
    """
    要去掉的字符替换成空格
    除了空白符可能还有其他
    :param para:
    :return:
    """
    para = re.sub("\s", " ", para)
    return para


def remove_space(para):
    """
    直接去掉字符
    除了空白符可能还有其他
    """
    para = re.sub("\s", "", para)
    return para


def cut_sent(para):
    """规则分句"""
    para = re.sub('([。！？!\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？!\?][”’])([^，。！？!\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    # para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


if __name__ == "__main__":
    para = "今天天气真好啊！！！很适合出去玩呢，你要和我一起嘛。。。走走走！ 不见不散"
    print(cut_sent(para))

# ori_text
# 添加换行符分句==添加又切分，无影响
# 先把所有要去掉的文字换成空格,位置和原句一一对应
# 然后再去掉空格做处理

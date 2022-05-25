# coding: utf-8

"""
@File    : preprocess.py.py
@Time    : 2022/4/19 10:52
@Author  : liuwangwang
@Software: PyCharm
"""


def is_chinese(usen):
    """判断一个unicode是否是汉字"""
    for uchar in usen:
        if '\u4e00' <= uchar <= '\u9fa5':
            continue
        else:
            return False
    else:
        return True

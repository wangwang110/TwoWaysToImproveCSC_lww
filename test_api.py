# coding: utf-8

"""
@File    : geyan_spider.py.py
@Time    : 2022/3/3 10:16
@Author  : liuwangwang
@Software: PyCharm
"""

import urllib.request
from lxml import etree
import os
import time
import re


def visit_xiaoxueurl():
    f = open("./ceshi.html").read()
    html = etree.HTML(f)
    res_li = []
    try:
        data_li = html.xpath('//p')
        text = ""
        for item in data_li:
            text += "\n"
            if item.text is not None:
                text += item.text
            elif text.strip() != "":
                res_li.append(text.strip())
                text = ""
            for sitem in item.findall("span"):
                if sitem.text is not None:
                    text += sitem.text
                    for ssitem in sitem.findall("br"):
                        if ssitem.tail is not None:
                            tmp = "\n" + ssitem.tail
                            text += tmp
            # print(text)
    except Exception as e:
        print(e)
    return res_li


if __name__ == "__main__":
    content_li = visit_xiaoxueurl()
    i = 1
    for s in content_li:
        print(i)
        print(s)
        print("#######")
        i += 1

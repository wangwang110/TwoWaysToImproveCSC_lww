# coding: utf-8

"""
@File    : geyan_spider.py.py
@Time    : 2022/3/3 10:16
@Author  : liuwangwang
@Software: PyCharm
"""

import json
import urllib.request
from lxml import etree
import os
import time
import re


def remove_ascii_control(text):
    """
    去除ascii中的控制符
    """
    text = re.sub("^\d+\.", "", text)
    text = re.sub("\s", "", text)
    return text.strip()


def visit_baikeurl():
    # 名人名言
    # 名言警句
    # 格言
    urls = [
        "https://baike.baidu.com/item/%E5%90%8D%E4%BA%BA%E5%90%8D%E8%A8%80/383132",
        "https://baike.baidu.com/item/%E5%90%8D%E8%A8%80%E8%AD%A6%E5%8F%A5/2622626",
        "https://baike.baidu.com/item/%E6%A0%BC%E8%A8%80/18073"
    ]
    res_geyan = []
    for url in urls:
        try:
            f = urllib.request.urlopen(url, timeout=5).read()
            html = etree.HTML(f)
            data_li = html.xpath('//div[@class="para"]')
            for item in data_li:
                sent = item.text
                if sent is None:
                    continue
                for sitem in item.findall("a"):
                    # print(item.find("a"))
                    sent += sitem.text
                    if sitem.tail is not None:
                        sent += sitem.tail
                ##
                if sent is not None:
                    res_geyan.append(sent)
        except Exception as e:
            print('url visit timeout: ', str(e))
            continue

    return res_geyan


if __name__ == "__main__":

    content_li = visit_baikeurl()
    final_res = []
    with open("./baike_mrmy.txt", "w", encoding="utf-8") as f:
        for item in content_li:
            item = remove_ascii_control(item)
            if "—" in item and 5 < len(item) <= 50:
                print(item)
                final_res.append(item)
                f.write(item + "\n")

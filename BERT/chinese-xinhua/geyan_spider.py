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


def remove_ascii_control(text):
    """
    去除ascii中的控制符
    """
    text = re.sub("\s", "", text)
    text = re.sub("\(.*?\)", "", text)
    text = re.sub("（.*?）", "", text)
    text = re.sub("\d+、", "", text)
    return text.strip()


def visit_xiaoxueurl():
    urls = [("https://www.sundxs.com/mingyan/690988.html", "小学各年级名人名言")]
    f = urllib.request.urlopen("https://www.lz13.cn/lizhi/mingrenmingyan.html", timeout=5).read()
    html = etree.HTML(f)
    for i in range(253, -1, -1):
        try:
            data_li = html.xpath('//div[@class="PostHead"]/span/h3/a')
            for data in data_li:
                print(data.text)
                if "小学" in data.text or "学生" in data.text or "年级" in data.text:
                    urls.append([data.attrib.get("href"), data.text])
            url = "https://www.lz13.cn/lizhi/mingrenmingyan-" + str(i) + ".html"
            f = urllib.request.urlopen(url, timeout=5).read()
            html = etree.HTML(f)
        except Exception as e:
            print(e)

    res_geyan = {}
    for item in urls:
        res_geyan_li = []
        sub_url, name = item
        print(name)
        time.sleep(3)
        sub_f = urllib.request.urlopen(sub_url, timeout=5).read()
        sub_html = etree.HTML(sub_f)
        sub_data_li = sub_html.xpath('//div[@class="PostContent"]/p')
        for item in sub_data_li:
            sent = item.text
            if sent is None:
                continue
            for sitem in item.findall("a"):
                sent += sitem.text
                if sitem.tail is not None:
                    sent += sitem.tail
            res_geyan_li.append(sent)
        res_geyan[name] = res_geyan_li
    return res_geyan


if __name__ == "__main__":
    content_dict = visit_xiaoxueurl()
    mrmy_set = set()
    with open("./xiaoxue_mrmy.txt", "w", encoding="utf-8") as fw:
        for name in content_dict:
            print(name)
            for text in content_dict[name]:
                text = re.sub("\u3000", "", text)
                text = re.sub("\u30001", "", text)
                if re.match("\d+、", text) is not None and re.search("[\u4e00-\u9fa5]", text) is not None \
                        and text not in mrmy_set:
                    text = remove_ascii_control(text)
                    fw.write(text + "\n")
                    mrmy_set.add(text)
                else:
                    print(text)

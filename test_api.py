# Copyright 2018 CVTE . All Rights Reserved.
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


# url = "https://poem.research-pro.sy.cvte.cn/poem_retrieve/"
# headers = {'content-type': 'application/json'}
#
# # 诗句
# requestData = {
#     "query": "天下谁人不识君", "intention": "诗句"
# }
# ret = requests.post(url, json=requestData, headers=headers)
# if ret.status_code == 200:
#     res = json.loads(ret.text)
#     print(res)
#
# # 题目
# requestData = {
#     "query": "咏鹅", "intention": "题目"
# }
# ret = requests.post(url, json=requestData, headers=headers)
# if ret.status_code == 200:
#     res = json.loads(ret.text)
#     print(res)
#
# # 题目
# requestData = {
#     "query": "咏梅", "intention": "题目","author_name":"陆游"
# }
# ret = requests.post(url, json=requestData, headers=headers)
# if ret.status_code == 200:
#     res = json.loads(ret.text)
#     print(res)
#
#
# # 作者
# requestData = {
#     "query": "白居易", "intention": "作者"
# }
# ret = requests.post(url, json=requestData, headers=headers)
# if ret.status_code == 200:
#     res = json.loads(ret.text)
#     print(res)

import requests
import json


url = "https://poem.research-pro.sy.cvte.cn/poem_retrieve/"
headers = {'content-type': 'application/json'}

# 诗句
requestData = {
    "query": "老骥伏枥", "intention": "诗句"
}
ret = requests.post(url, json=requestData, headers=headers)
if ret.status_code == 200:
    res = json.loads(ret.text)
    print(res)

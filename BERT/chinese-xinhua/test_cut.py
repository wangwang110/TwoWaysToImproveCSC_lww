#
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json

# 由于MD5模块在python3中被移除
# 在python3中使用hashlib模块进行md5操作

import hashlib

# 待加密信息
str = ""
# 创建md5对象
hl = hashlib.md5()

APP_ID = "5bfd6c7ae85147da829b28a108c690c1APP"
Secret = "mF2RoGogHwKKGDUHD8uUrGZkOUmmCOX3"
post_dict = {
    "x-sw-app-id": "5bfd6c7ae85147da829b28a108c690c1",
    # "x-sw-content-md5": "50A84318288796E1417117057B003050",
    "x-sw-req-path": "/discipline-tool/english/gec",
    "x-sw-sign-type": "md5",
    "x-sw-timestamp": "1520995583906",
    "x-sw-version": "2",
}
str = Secret
item_li = sorted(post_dict.items(), key=lambda s: s[0], reverse=False)
print(item_li)
for item in item_li:
    str += item[0]
    str += item[1]
str += Secret
print(str)

# Tips
# 此处必须声明encode
# 若写法为hl.update(str)  报错为： Unicode-objects must be encoded before hashing
hl.update(str.encode(encoding='utf-8'))

print('MD5加密前为 ：' + str)
print('MD5加密后为 ：' + hl.hexdigest())

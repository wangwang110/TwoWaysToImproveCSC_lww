# coding: utf-8

import json
import hashlib

body = {
            "article":"I lvve bu====y some cakes",
            "spell_detect": 0
        }

str = json.dumps(body)
m = hashlib.md5()
m.update(str.encode("utf8"))
print(m.hexdigest())
if m.hexdigest() == "BE4C53295E7834193C510E764F686EF8":
    print("==========")
# BE4C53295E7834193C510E764F686EF8

#
# APP_ID = "5bfd6c7ae85147da829b28a108c690c1"
# Secret = "mF2RoGogHwKKGDUHD8uUrGZkOUmmCOX3"
# post_dict = {
#     "x-sw-app-id": APP_ID,
#     # "x-sw-content-md5": "50A84318288796E1417117057B003050",
#     "x-sw-req-path": "/discipline-tool/english/gec",
#     "x-sw-sign-type": "md5",
#     "x-sw-timestamp": "1520995583906",
#     "x-sw-version": "2",
# }
#
# str = Secret
# item_li = sorted(post_dict.items(), key=lambda s: s[0], reverse=False)
# print(item_li)
# for item in item_li:
#     str += item[0]
#     str += item[1]
# str += Secret
# print(str)
#
# m = hashlib.md5()
# m.update(str.encode("utf8"))
# print(m.hexdigest())
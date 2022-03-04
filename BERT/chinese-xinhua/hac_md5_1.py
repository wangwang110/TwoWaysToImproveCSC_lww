# coding: utf-8

import json
import hashlib

body = {
    "article": "I lvve bu====y some cakes",
    "spell_detect": 0
}

str = json.dumps(body)
m = hashlib.md5()
m.update(str.encode("utf8"))
print(m.hexdigest())
if m.hexdigest() == "BE4C53295E7834193C510E764F686EF8":
    print("==========")
# BE4C53295E7834193C510E764F686EF8

import hmac
import hashlib

# 第一个参数是密钥key，第二个参数是待加密的字符串，第三个参数是hash函数
mac = hmac.new(bytes('mF2RoGogHwKKGDUHD8uUrGZkOUmmCOX3', encoding="utf-8"), bytes(str, encoding="utf-8"), hashlib.md5)
mac.digest()  # 字符串的ascii格式
print(mac.hexdigest())  # 加密后字符串的十六进制格式

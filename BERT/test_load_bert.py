import torch
import torch.nn as nn

import torch
import torch.nn as nn
import numpy as np
import math


def is_chinese(usen):
    """判断一个unicode是否是汉字"""
    for uchar in usen:
        if uchar >= '\u4e00' and uchar <= '\u9fa5':
            continue
        else:
            return False
    else:
        return True


# bert的词典
vob2id = {}
with open("/data_local/plm_models/chinese_L-12_H-768_A-12/vocab.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        vob2id.setdefault(line.strip(), i)
        tag = 1
        if len(line.strip()) >= 2 and not line.startswith("##"):
            tag = 0
            for t in line.strip():
                if not is_chinese(t):
                    tag = 2
                    break
        if tag == 0:
            print(line)

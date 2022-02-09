# -*- coding: utf-8 -*-
"""
Created on Tue Dec 05 15:21:42 2017

@author: lww
"""

import numpy as np
import math
import torch


# 原单词不再前10，才认为有错
sorted, indices = torch.sort(out_prob, dim=-1, descending=True)
indices = indices[:, :, :5]
out_new = copy.deepcopy(input_ids)
for i in range(len(out)):
    for j in range(input_lens[i]):
        if out[i][j] != input_ids[i][j] and input_ids[i][j] not in indices[i][j]:
            out_new[i][j] = out[i][j]
out = out_new

# 检测有错，并且修正了的才算真正的有错

# out_new = copy.deepcopy(input_ids)
# for i in range(len(out)):
#     for j in range(input_lens[i]):
#         if torken_prob[i][j] >= 0.5 and out[i][j] != input_ids[i][j]:
#             out_new[i][j] = out[i][j]
#
# out = out_new
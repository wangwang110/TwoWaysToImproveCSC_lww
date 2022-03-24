# coding: utf-8

"""
@File    : test.py.py
@Time    : 2022/3/16 18:44
@Author  : liuwangwang
@Software: PyCharm
"""
import torch

model_path = "/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_all/sighan13/model.pkl"
res_dict = torch.load(model_path)
print(res_dict)

# import torch
# import torch.nn as nn
#
# x = torch.FloatTensor([[0.1, 0.2, 0.4, 0.8], [0.1, 0.2, 0.4, 0.8]])
# print(x.size())
# y = torch.LongTensor([3, 3])
# print(y.size())
#
# loss = nn.MultiMarginLoss(reduction="none")
# loss_val = loss(x, y)
# print(loss_val)
#
# loss = nn.MultiMarginLoss(reduction="sum")
# loss_val = loss(x, y)
# print(loss_val.item())
# print(loss_val.item() / x.size(0))
# # 验证
# print(1 / 2 * 1 / 4 * ((1 - 0.8 + 0.1) + (1 - 0.8 + 0.2) + (1 - 0.8 + 0.4) +
#                        (1 - 0.8 + 0.1) + (1 - 0.8 + 0.2) + (1 - 0.8 + 0.4)))

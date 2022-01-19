# -*- coding：utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import time
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn import CrossEntropyLoss
from focalloss import *
import numpy as np


def setup_seed(seed):
    # set seed for CPU
    torch.manual_seed(seed)
    # set seed for current GPU
    torch.cuda.manual_seed(seed)
    # set seed for all GPU
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    # Cancel acceleration
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)


setup_seed(20)


# 定义模型
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_out):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.out = nn.Linear(n_hidden, n_out)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.hidden.weight.data.uniform_(-initrange, initrange)
        self.hidden.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()

    def forward(self, x, y=None):
        x = self.hidden(x)
        x = torch.sigmoid(x)
        x = self.out(x)
        out = F.log_softmax(x, dim=1)
        return out


# 构造数据
data_x = [torch.randn(32, 50)] * 16
data_y = [[1 if random.random() > 0.5 else 0 for j in range(32)]] * 16

# 模型
net = Net(n_feature=50, n_hidden=10, n_out=2)

# 优化器
optimizer = optim.Adam(net.parameters(), lr=1e-3)

loss_fct = CrossEntropyLoss()
loss_fct_focal = FocalLoss(gamma=1, alpha=[1, 0.8])

for epoch in range(10):
    for step, batch in enumerate(zip(data_x, data_y)):
        x, y = batch
        y = torch.tensor(y)
        out = net(x, y)
        loss = loss_fct_focal(out, y)
        loss.backward()
        print(loss.item())
        optimizer.step()
        optimizer.zero_grad()

# start_time = time.time()
# device = "cpu"
# batch = 1024
# num_class = 2
# maxe = 0
# for i in range(1000):
#     x = torch.rand(batch, num_class).to(device)
#     l = torch.rand(batch).ge(0.1).long().to(device)
#
#     output0 = FocalLoss(gamma=0)(x, l)
#     output1 = nn.CrossEntropyLoss()(x, l)
#     a = output0.item()
#     b = output1.item()
#     if abs(a - b) > maxe:
#         maxe = abs(a - b)
# print('time:', time.time() - start_time, 'max_error:', maxe)
#
# start_time = time.time()
# maxe = 0
# for i in range(100):
#     x = torch.rand(batch, 1000, 8, 4) * random.randint(1, 10)
#     x = Variable(x.cuda())
#     l = torch.rand(batch, 8, 4) * 1000  # 1000 is classes_num
#     l = l.long()
#     l = Variable(l.cuda())
#
#     output0 = FocalLoss(gamma=0)(x, l)
#     output1 = nn.NLLLoss2d()(F.log_softmax(x), l)
#     a = output0.data[0]
#     b = output1.data[0]
#     if abs(a - b) > maxe: maxe = abs(a - b)
#
# print('time:', time.time() - start_time, 'max_error:', maxe)

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import numpy as np
import math

# m = nn.BatchNorm2d(2, affine=False)
# x = [[[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]], [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]],
#      [[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]], [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]]]
#
# input = torch.tensor(x, dtype=torch.float)
# print(input.size())
# output = m(input)
# print(output)
#
# y = [[[[1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6]], [[1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6]]],
#      [[[1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6]], [[1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6]]]]
# input = torch.tensor(y, dtype=torch.float)
# print(input.size())
# output = m(input)
# print(output)

z = [[[[1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6]]]]
# 1*1*12
input = torch.tensor(z, dtype=torch.float)
print(input.size())

batch_norm = nn.BatchNorm2d(1, affine=False)
batch_out = batch_norm(input)
print(batch_out)

layer_norm = nn.LayerNorm(12)
layer_out = layer_norm(input)
print(layer_out)


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


setup_seed(10)
for j in range(10):
    for i in range(1):
        print(np.random.randint(0, 100))

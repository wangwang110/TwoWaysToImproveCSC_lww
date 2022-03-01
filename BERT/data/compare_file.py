#
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import filecmp

# 如果两边路径的头文件都存在，进行比较
try:
    status = filecmp.cmp("./autog_wang_27w.txt", "./autog_wang_tt.txt")
    # 为True表示两文件相同
    if status:
        print("files are the same")
    # 为False表示文件不相同
    else:
        print("files are different")
# 如果两边路径头文件不都存在，抛异常
except IOError:
    print("Error:" + "File not found or failed to read")

# 如果两边路径的头文件都存在，进行比较
try:
    status = filecmp.cmp("./autog_wang_27w.txt", "./autog_wang_st.txt")
    # 为True表示两文件相同
    if status:
        print("files are the same")
    # 为False表示文件不相同
    else:
        print("files are different")
# 如果两边路径头文件不都存在，抛异常
except IOError:
    print("Error:" + "File not found or failed to read")

# 如果两边路径的头文件都存在，进行比较
try:
    status = filecmp.cmp("./punct_data/rep_autog_wang_train.txt", "./rep_autog_wang_train.txt")
    # 为True表示两文件相同
    if status:
        print("files are the same")
    # 为False表示文件不相同
    else:
        print("files are different")
# 如果两边路径头文件不都存在，抛异常
except IOError:
    print("Error:" + "File not found or failed to read")

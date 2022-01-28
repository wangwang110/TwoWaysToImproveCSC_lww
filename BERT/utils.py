# -*- coding: utf-8 -*-
"""
Created on Tue Dec 05 15:21:42 2017

@author: lww
"""

import numpy as np
import math


def cal_fuzzy(output):
    m, n = np.shape(np.array(output))
    # m个样本n类

    fuzzy = []
    fuzzy_y = []
    for i in range(m):
        temp = 0
        fuzzy_y.append(np.argmax(output[i]))
        for j in range(n):
            temp += output[i][j] * math.log(output[i][j]) + (1.0 - output[i][j]) * math.log(1.0 - output[i][j])

        temp = -1.0 / n * temp
        # print temp
        # 越小，清晰度越高
        fuzzy.append(temp)
    ##加入清晰度高的样本，和预测的标签
    return [fuzzy, fuzzy_y]

# a=np.array([[0.1,0.9],[0.2,0.8],[0.3,0.7],[0.4,0.6],[0.5,0.5],[0.6,0.4],[0.7,0.3],[0.8,0.2],[0.9,0.1]])
# fuzzy,fuzzy_y= cal_fuzzy(a)
# outfile=open('fuzzy.txt','w')
# for f in sorted(fuzzy):
#    outfile.write(str(f))
#    outfile.writelines('\n')
# outfile.close()

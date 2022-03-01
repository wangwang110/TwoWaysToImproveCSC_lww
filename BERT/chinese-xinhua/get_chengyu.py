#
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json

path = "/data_local/TwoWaysToImproveCSC/BERT/chinese-xinhua/data/ci.json"
with open(path, 'r', encoding="utf-8") as f:
    chengyu_dict = json.load(f)
    print(len(chengyu_dict))

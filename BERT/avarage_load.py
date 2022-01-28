#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

models = []
for i in [20, 30, 40, 50, 60]:
    model_path = "/data_local/TwoWaysToImproveCSC/BERT/save/baseline/initial/sighan13_" + str(i) + "/model.pkl"
    models.append(model_path)

#

base_model_path = "/data_local/TwoWaysToImproveCSC/BERT/save/baseline/sighan13/model.pkl"
checkpoint_path = os.path.join(base_model_path)
state = torch.load(checkpoint_path)

count = 0
for cpt in models:
    count += 1
    tmp_state = torch.load(cpt)
    for k in tmp_state:
        state[k] += tmp_state[k]
for k in state:
    state[k] = state[k] / (count + 1)

dir_path = "/data_local/TwoWaysToImproveCSC/BERT/save/baseline/initial/sighan13_avg/"
new_checkpoint_path = dir_path + "/model_avg.pkl"
torch.save(state, new_checkpoint_path)
print(state)

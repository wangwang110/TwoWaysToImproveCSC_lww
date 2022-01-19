# -*- coding: UTF-8 -*-

import sys

sys.path.append("..")
import torch.nn as nn
import torch
import re
import numpy as np
import argparse
from transformers import BertModel, BertConfig, BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import operator
from model import li_testconstruct, BertDataset, BFTLogitGen, readAllConfusionSet, cc_testconstruct, construct
import os

vob = {}
with open("/data_local/plm_models/chinese_L-12_H-768_A-12/vocab.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        vob.setdefault(i, line.strip())

b2v = {v: k for k, v in vob.items()}


class BertMlm:
    def __init__(self, bert_path):
        """
        :param bert_path:
        :param model_path:
        """

        bert = BertForMaskedLM.from_pretrained(bert_path, return_dict=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.config = BertConfig.from_pretrained(bert_path)
        self.batch_size = 16
        self.model = bert.to(self.device)
        # self.confusion_set = readAllConfusionSet('/data_local/TwoWaysToImproveCSC/BERT/save/confusion.file')
        # /data_local/TwoWaysToImproveCSC/BERT/save/confusion.file
        # ../save/confusion.file

    def test_without_trg(self, all_texts):
        self.model.eval()
        test = li_testconstruct(all_texts)
        test = BertDataset(self.tokenizer, test)
        test = DataLoader(test, batch_size=int(self.batch_size), shuffle=False)
        model_predict_ids = []
        for batch in test:
            inputs = self.tokenizer(batch['input'], padding=True, truncation=True, return_tensors="pt").to(self.device)
            max_len = 180
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'][:, :max_len], \
                                                    inputs['token_type_ids'][:, :max_len], \
                                                    inputs['attention_mask'][:, :max_len]

            # input_lens = torch.sum(input_attn_mask, 1)
            model_out = self.model(input_ids, input_tyi, input_attn_mask)
            out = model_out.logits
            # batch * seq_len * vocab_size
            # torch.gather

            # out_id_sort = [p for p in torch.argsort(out, dim=2, descending=True)]
            out_logit = [s for s in out.detach().cpu().numpy()]

            model_predict_ids.extend(out_logit)

            # # 返回修改后的句子
            # out = out.argmax(dim=-1)
            # # 修改过的句子
            # res = []
            # num_sample, _ = input_ids.size()
            # for idx in range(num_sample):
            #     sent_li = []
            #     for t in range(1, input_lens[idx] - 1):
            #         if vob[out[idx][t].item()] == "[UNK]":
            #             sent_li.append(vob[input_ids[idx][t].item()])
            #         else:
            #             sent_li.append(vob[out[idx][t].item()])
            #     res.append("".join(sent_li))
            #     print(res)
        return model_predict_ids


if __name__ == "__main__":
    # add arguments
    parser = argparse.ArgumentParser(description="choose which model")
    parser.add_argument('--task_name', type=str, default='bert_pretrain')
    parser.add_argument('--load_path', type=str, default='./save/13_train_seed0_1.pkl')

    # parser.add_argument('--batch_size', type=int, default=20)
    # parser.add_argument('--epoch', type=int, default=1)
    # parser.add_argument('--learning_rate', type=float, default=2e-5)
    # parser.add_argument('--do_save', type=str2bool, nargs='?', const=False)
    # parser.add_argument('--save_dir', type=str, default='../save')
    # parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    task_name = args.task_name
    print("----Task: " + task_name + " begin !----")

    # 初始化模型
    bert_path = "/data_local/plm_models/chinese_L-12_H-768_A-12/"
    # args.load_path = "/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/sighan13/epoch10.pkl"
    obj = BertMlm(bert_path)
    model_ouput = obj.test_without_trg(["遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。"])
    print(model_ouput)

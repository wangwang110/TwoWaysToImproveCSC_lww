# -*- coding: UTF-8 -*-

import sys

sys.path.append("..")
import torch.nn as nn
import torch
import re
import numpy as np
import argparse
from transformers import BertModel, BertConfig, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import operator
from model import BertCSC, li_testconstruct, BertDataset, BFTLogitGen, readAllConfusionSet, cc_testconstruct, construct
import os

vob = {}
with open("/data_local/plm_models/chinese_L-12_H-768_A-12/vocab.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        vob.setdefault(i, line.strip())


class CSCmodel:
    def __init__(self, bert_path, model_path):
        """
        :param bert_path:
        :param model_path:
        """

        bert = BertModel.from_pretrained(bert_path, return_dict=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.config = BertConfig.from_pretrained(bert_path)
        self.batch_size = 20
        self.model = BertCSC(bert, self.tokenizer, self.device).to(self.device)

        self.model.load_state_dict(torch.load(model_path))

        self.confusion_set = readAllConfusionSet('/data_local/TwoWaysToImproveCSC/BERT/save/confusion.file')
        # /data_local/TwoWaysToImproveCSC/BERT/save/confusion.file
        # ../save/confusion.file

    def test_without_trg(self, all_texts):
        self.model.eval()
        test = li_testconstruct(all_texts)
        test = BertDataset(self.tokenizer, test)
        test = DataLoader(test, batch_size=int(self.batch_size), shuffle=False)
        res = []
        srcs = []
        for batch in test:
            inputs = self.tokenizer(batch['input'], padding=True, truncation=True, return_tensors="pt").to(self.device)
            max_len = 180
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'][:, :max_len], \
                                                    inputs['token_type_ids'][:, :max_len], \
                                                    inputs['attention_mask'][:, :max_len]

            input_lens = torch.sum(input_attn_mask, 1)
            out = self.model(input_ids, input_tyi, input_attn_mask)
            # 分析预测的前n个
            # out_id_sort = [p for p in torch.argsort(out, dim=2, descending=True)[:, :, :10]]
            #
            # res_analysis = []
            # sent_li = []
            # for t in range(1, input_lens[0] - 1):
            #     tmp_li = []
            #     for s in range(10):
            #         tmp_li.append(vob[out_id_sort[0][t][s].item()])
            #     res_analysis.append(tmp_li)
            # print(res_analysis)

            # 返回修改后的句子
            out = out.argmax(dim=-1)
            # 修改过的句子

            num_sample, _ = input_ids.size()
            for idx in range(num_sample):
                sent_li = []
                src_sent_li = []
                for t in range(1, input_lens[idx] - 1):
                    src_sent_li.append(vob[input_ids[idx][t].item()])
                    if vob[out[idx][t].item()] == "[UNK]":
                        sent_li.append(vob[input_ids[idx][t].item()])
                    else:
                        sent_li.append(vob[out[idx][t].item()])
                res.append("".join(sent_li))
                srcs.append("".join(src_sent_li))

        return srcs, res


def correct_file(path, path_out):
    with open(path, "r", encoding="utf-8") as f, open(path_out, "w", encoding="utf-8") as fw:
        all_texts = []
        try:
            for line in f.readlines():
                key, src = line.strip().split(" ")
                all_texts.append(src)
        except Exception as e:
            print(e)

        all_texts_src = sorted(all_texts, key=lambda s: len(s))
        all_srcs, all_trgs = obj.test_without_trg(all_texts_src)
        for text, s, t in zip(all_texts_src, all_srcs, all_trgs):
            if s == t:
                fw.write(text + "\n")


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
    args.load_path = "/data_local/TwoWaysToImproveCSC/BERT/save/bert_paper_model/preTrain/sighan13/model.pkl"
    # args.load_path = "/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/sighan13/model.pkl"
    # /data_local/TwoWaysToImproveCSC/BERT/save/bert_paper_model/preTrain/sighan13
    #  "/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/sighan13/epoch10.pkl"
    obj = CSCmodel(bert_path, args.load_path)
    input_text, model_ouput = obj.test_without_trg([
        "在我的印象中，哑巴是不同于正常人的，他说话总是叽里呱啦的，让人难解；我也常听别人说，哑巴非常“狠”，常打或者吓唬小孩，因此我从记事起就对哑巴产生了一种恐惧感，总怕突然被哑巴抓住暴打一顿，所以，每次上学经过哑巴叔叔门口时，总是跺的远远的。 "])
    print(input_text[0])
    print(model_ouput[0])

    correct_file("./cc_data/xiaoxue_sent_all.txt", "./cc_data/xiaoxue_sent_all_cor.txt")
# -*- coding: UTF-8 -*-

import sys

sys.path.append("..")
import numpy as np
import os
import torch
import argparse
from transformers import BertForMaskedLM, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
from csc_utils import cut_sent
from model import BertFineTuneMac, BertCSC, li_testconstruct, BertDataset, BFTLogitGen, readAllConfusionSet, \
    cc_testconstruct, construct


class CSCmodel:
    def __init__(self, bert_path, model_path, gpu_id="6"):
        """
        :param bert_path:
        :param model_path:
        :param gpu_id:
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        bert = BertForMaskedLM.from_pretrained(bert_path, return_dict=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.config = BertConfig.from_pretrained(bert_path)
        self.batch_size = 20
        device_ids = []
        self.model = BertFineTuneMac(bert, self.tokenizer, self.device, device_ids, is_correct_sent=True).to(
            self.device)
        # device, device_ids, is_correct_sent=True
        # param_dict = {k.replace("bert.bert.", "bert."): v for k, v in torch.load(model_path).items()}
        self.model.load_state_dict(torch.load(model_path))
        # 混淆集
        self.confusion_set = readAllConfusionSet('/data_local/TwoWaysToImproveCSC/BERT/save/confusion.file')
        # bert的词典
        self.vob = {}
        with open("/data_local/plm_models/chinese_L-12_H-768_A-12/vocab.txt", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.vob.setdefault(i, line.strip())

    def test_without_trg(self, all_texts, correct_tag=False):
        self.model.eval()
        test = li_testconstruct(all_texts)
        test = BertDataset(test)
        test = DataLoader(test, batch_size=int(self.batch_size), shuffle=False)
        res = []
        srcs = []
        for batch in test:
            inputs = self.help_vectorize(batch['input'])
            max_len = 180
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'][:, :max_len], \
                                                    inputs['token_type_ids'][:, :max_len], \
                                                    inputs['attention_mask'][:, :max_len]

            input_lens = torch.sum(input_attn_mask, 1)
            out = self.model(input_ids, input_tyi, input_attn_mask)
            out = out[-1].argmax(dim=-1)
            num_sample, _ = input_ids.size()
            # 修改过的句子
            if correct_tag:
                for idx in range(num_sample):
                    ori_ids = input_ids[idx][1:input_lens[idx] - 1]
                    pre_ids = out[idx][1:input_lens[idx] - 1]
                    tag = 0
                    for s, t in zip(ori_ids, pre_ids):
                        if s.item() != t.item() and t.item() != 100:
                            tag = 1
                            break
                    if tag == 0:
                        res.append(batch['input'][idx])
                        srcs.append(batch['input'][idx])
            else:
                num_sample, _ = input_ids.size()
                for idx in range(num_sample):
                    sent_li = []
                    src_sent_li = []
                    for t in range(1, input_lens[idx] - 1):
                        src_sent_li.append(self.vob[input_ids[idx][t].item()])
                        if self.vob[out[idx][t].item()] == "[UNK]":
                            sent_li.append(self.vob[input_ids[idx][t].item()])
                        else:
                            sent_li.append(self.vob[out[idx][t].item()])
                    res.append("".join(sent_li))
                    srcs.append("".join(src_sent_li))

        return srcs, res

    def help_vectorize(self, src_li):
        """
        :param batch:
        :return:
        """
        max_seq_length = max([len(src) for src in src_li]) + 2
        inputs = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}

        for src in src_li:
            input_ids, input_mask, segment_ids = self.text2vec(src, max_seq_length)
            inputs['input_ids'].append(input_ids)
            inputs['token_type_ids'].append(segment_ids)
            inputs['attention_mask'].append(input_mask)

        inputs['input_ids'] = torch.tensor(np.array(inputs['input_ids'])).to(self.device)
        inputs['token_type_ids'] = torch.tensor(np.array(inputs['token_type_ids'])).to(self.device)
        inputs['attention_mask'] = torch.tensor(np.array(inputs['attention_mask'])).to(self.device)

        return inputs

    def text2vec(self, src, max_seq_length):
        """
        :param src:
        :return:
        """
        tokens_a = [a for a in src]
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for j, tok in enumerate(tokens):
            if tok not in self.tokenizer.vocab:
                tokens[j] = "[UNK]"

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        return input_ids, input_mask, segment_ids


def correct_file(path, path_out):
    with open(path, "r", encoding="utf-8") as f, open(path_out, "w", encoding="utf-8") as fw:
        all_texts = []
        try:
            for line in f.readlines():
                item = line.strip().split(" ")
                if len(item) == 2:
                    key, src = item
                    all_texts.append(src)
                elif len(item) == 1:
                    src = item[0]
                    all_texts.append(src)
        except Exception as e:
            print(e)

        all_texts_src = sorted(all_texts, key=lambda s: len(s))
        all_srcs, all_trgs = obj.test_without_trg(all_texts_src, correct_tag=True)
        for s, t in zip(all_srcs, all_trgs):
            if s == t:
                fw.write(s + "\n")


if __name__ == "__main__":
    # add arguments
    parser = argparse.ArgumentParser(description="choose which model")
    parser.add_argument('--task_name', type=str, default='bert_pretrain')
    # parser.add_argument('--load_path', type=str, default='./save/13_train_seed0_1.pkl')

    # 处理预训练数据
    parser.add_argument('--input', type=str, default="./data/13train_trg.txt")
    parser.add_argument('--output', type=str, default="./data/tmp.txt")

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
    # load_path = "/data_local/TwoWaysToImproveCSC/BERT/save/bert_paper_model/preTrain/sighan13/model.pkl"
    load_path = "/data_local/TwoWaysToImproveCSC/BERT/save/bert_paper_model/sighan13/model_new.pkl"
    # args.load_path = "/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/sighan13/model.pkl"
    # /data_local/TwoWaysToImproveCSC/BERT/save/bert_paper_model/preTrain/sighan13
    #  "/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/sighan13/epoch10.pkl"
    obj = CSCmodel(bert_path, load_path)
    input_text, model_ouput = obj.test_without_trg([
        "布告栏转眼之间从不起眼的丑小鸭变成了高贵优雅的天鹅！仅管这大改造没有得名，但过程也是很可贵的。",
        "我爱北进天安门",
        "我爱北京天按门",
        "没过几分钟，救护车来了，发出响亮而清翠的声音",
        "我见过一望无际、波澜壮阔的大海；玩赏过水平如镜、诗情画意的西湖；游览过翡翠般的漓江；让我难以忘怀的要数那荷叶飘香、群山坏绕的普者黑。"])
    # print(input_text)
    print(model_ouput)

    # 与前后无法组成词就不改

    with open("./data/13test.txt", "r", encoding="utf-8") as f, \
            open("./data/13test.pre", "w", encoding="utf-8") as fw:
        all_texts = []
        for line in f.readlines():
            try:
                src, trg = line.strip().split(" ")
                all_texts.append(src)
            except:
                continue

        input_text, model_ouputs = obj.test_without_trg(all_texts)
        for src, trg in zip(all_texts, model_ouputs):
            fw.write(src + " " + trg + "\n")

    with open("./data/chinese_spell_4.txt", "r", encoding="utf-8") as f, \
            open("./data/chinese_spell_4.pre", "w", encoding="utf-8") as fw:
        all_texts = []
        for line in f.readlines():
            try:
                src, trg = line.strip().split(" ")
                all_texts.append(src)
            except:
                continue

        input_text, model_ouputs = obj.test_without_trg(all_texts)
        for src, trg in zip(all_texts, model_ouputs):
            fw.write(src + " " + trg + "\n")

    correct_file(args.input, args.output)
    # 微博预训练的语料，要先过一遍纠错模型（可能存在很多错误）

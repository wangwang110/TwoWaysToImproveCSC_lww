# -*- coding: UTF-8 -*-

import sys

sys.path.append("..")
import torch
import argparse
from transformers import BertModel, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
from model import BertCSC, li_testconstruct, BertDataset, BFTLogitGen, readAllConfusionSet, cc_testconstruct, construct

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

    def test(self, data):
        sent = data["input"]
        self.model.eval()
        test = li_testconstruct([sent])
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
        all_srcs, all_trgs = obj.test_without_trg(all_texts_src)
        for text, s, t in zip(all_texts_src, all_srcs, all_trgs):
            if s == t:
                fw.write(text + "\n")


if __name__ == "__main__":
    # add arguments
    parser = argparse.ArgumentParser(description="choose which model")
    parser.add_argument('--task_name', type=str, default='bert_pretrain')
    # parser.add_argument('--load_path', type=str, default='./save/13_train_seed0_1.pkl')

    # 处理预训练数据
    parser.add_argument('--input', type=str, default="./cc_data/xiaoxue_sent_all.txt")
    parser.add_argument('--output', type=str, default="./cc_data/xiaoxue_sent_all_tmp.txt")

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
    load_path = "/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998/sighan13/model.pkl"
    # args.load_path = "/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/sighan13/model.pkl"
    # /data_local/TwoWaysToImproveCSC/BERT/save/bert_paper_model/preTrain/sighan13
    #  "/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/sighan13/epoch10.pkl"
    obj = CSCmodel(bert_path, load_path)
    input_text, model_ouput = obj.test_without_trg([
        "没过几分钟，救护车来了，发出响亮而清翠的声音",
        "我见过一望无际、波澜壮阔的大海；玩赏过水平如镜、诗情画意的西湖；游览过翡翠般的漓江；让我难以忘怀的要数那荷叶飘香、群山都坏绕的普者黑。"])
    print(input_text)
    print(model_ouput)

    # correct_file(args.input, args.output)
    # 微博预训练的语料，要先过一遍纠错模型（可能存在很多错误）

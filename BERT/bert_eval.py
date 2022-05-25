# -*- coding: UTF-8 -*-

import sys

sys.path.append("..")
import time
import torch
import re
import numpy as np
from optparse import OptionParser
from transformers import BertModel, BertConfig, BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset
from model import li_testconstruct, BertDataset, BFTLogitGen, readAllConfusionSet, cc_testconstruct, construct
import pickle
from tqdm import tqdm

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
        self.batch_size = 25
        self.model = bert.to(self.device)
        self.model.eval()
        tmp_dict = readAllConfusionSet('/data_local/TwoWaysToImproveCSC/BERT/save/confusion.file')

        self.confusion_set = {}
        # self.confusion_set_unk = {}
        for key in tmp_dict:
            self.confusion_set[key] = [s for s in tmp_dict[key]]
            # 如果不b2v就会被认为是unk，修改unk位置是否被认为合理
            # 修改unk位置应该非常常见
            # unk 说明位置很有可能不对
            # 改成unk是没有用的
            # self.confusion_set_unk = [s for s in tmp_dict[key] if s not in b2v]
        self.confusion_set_id = {}
        for key in self.confusion_set:
            item_li = []
            for s in self.confusion_set[key]:
                if s in b2v:
                    item_li.append(b2v[s])
                else:
                    item_li.append(100)
            self.confusion_set_id[key] = item_li

    def test_without_trg(self, all_texts, vocab):
        test = li_testconstruct(all_texts)
        test = BertDataset(self.tokenizer, test)
        test = DataLoader(test, batch_size=int(self.batch_size), shuffle=False)

        for batch in test:
            inputs = self.tokenizer(batch['input'], padding=True, truncation=True, return_tensors="pt").to(
                self.device)
            max_len = 180
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'][:, :max_len], \
                                                    inputs['token_type_ids'][:, :max_len], \
                                                    inputs['attention_mask'][:, :max_len]
            model_out = self.model(input_ids, input_tyi, input_attn_mask)
            out = model_out.logits
            #
            input_lens = torch.sum(input_attn_mask, 1).detach().cpu().numpy()
            input_ids_li = [s for s in input_ids.detach().cpu().numpy()]
            out_logit = [s for s in out.detach().cpu().numpy()]
            batch_size, _ = input_ids.size()
            batch_res = []
            for sent_id in range(batch_size):
                num = input_lens[sent_id] - 2  # [cls],[sep]
                tokens = [self.tokenizer.ids_to_tokens[t] for t in input_ids_li[sent_id][1:num + 1]]
                # tokens = self.tokenizer.tokenize(batch['input'][sent_id])
                # self.tokenizer.tokenize()

                index_li = [s for s in range(num)]
                up_num = num // 4
                np.random.shuffle(index_li)
                count = 0

                for i in index_li:
                    if tokens[i] in self.confusion_set:
                        count += 1
                        if np.random.rand(1) > 0.9:
                            # 随机替换为词表中的词
                            idx = np.random.randint(0, len(vocab))
                            tokens[i] = vocab[idx]
                        else:
                            # 替换为候选集中，最有可能出错的词，bert概率输出最大的，并且在候选集中的词
                            token_conf = self.confusion_set[tokens[i]]
                            token_conf_id = self.confusion_set_id[tokens[i]]
                            id2prob = [out_logit[sent_id][i][j] for j in token_conf_id]
                            idx = np.argmax(id2prob).item()
                            tokens[i] = token_conf[idx]
                        if count == up_num:
                            break

                batch_res.append("".join(tokens).replace("##", "") + " " + batch['input'][sent_id])
            yield batch_res

    def mlm_predict(self, all_texts):
        test = li_testconstruct(all_texts)
        test = BertDataset(self.tokenizer, test)
        test = DataLoader(test, batch_size=int(self.batch_size), shuffle=False)

        for batch in test:
            inputs = self.tokenizer(batch['input'], padding=True, truncation=True, return_tensors="pt").to(
                self.device)
            max_len = 180
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'][:, :max_len], \
                                                    inputs['token_type_ids'][:, :max_len], \
                                                    inputs['attention_mask'][:, :max_len]
            model_out = self.model(input_ids, input_tyi, input_attn_mask)
            out = model_out.logits
            #
            input_lens = torch.sum(input_attn_mask, 1).detach().cpu().numpy()
            prob, index = torch.max(out, 2)
            # prob, index = torch.argsort(out, dim=2, descending=True)[:, :, :10]

            batch_size, _ = input_ids.size()
            batch_res = []
            for sent_id in range(batch_size):
                num = input_lens[sent_id] - 2  # [cls],[sep]
                tokens = [self.tokenizer.ids_to_tokens[t.item()] for t in index[sent_id][1:num + 1]]
                batch_res.append("".join(tokens).replace("##", ""))
                print("".join(tokens).replace("##", ""))

    def read_path(self, path):
        print("reading now......")
        all_texts = []
        true_texts = []
        with open(path, encoding="UTF-8") as f:
            for line in f.readlines():
                if 6 < len(line) < 160 and not line.startswith("）"):
                    if len(set(line) & set(self.confusion_set.keys())) != 0:
                        all_texts.append(line.strip().lower())
                    else:
                        true_texts.append(line.strip().lower())
        return all_texts, true_texts


def GenerateCSC(infile, outfile, vocab):
    """
    :param input_path:
    :param output_path:
    :param vocab_path:
    :return:
    """
    bert_path = "/data_local/plm_models/chinese_L-12_H-768_A-12/"
    obj = BertMlm(bert_path)
    all_texts, true_texts = obj.read_path(infile)

    sort_all_texts = sorted(all_texts, key=lambda s: len(s))
    # 长度越短一个batch应该可以处理的越多

    all_texts_li = obj.test_without_trg(sort_all_texts, vocab)

    t0 = time.time()
    with open(outfile, "w", encoding="utf-8") as fw:
        for texts in tqdm(all_texts_li):
            for text in texts:
                fw.write(text + "\n")
        for text in true_texts:
            fw.write(text + " " + text + "\n")

    print(time.time() - t0)


if __name__ == "__main__":
    # parser = OptionParser()
    # parser.add_option("--path", dest="path", default="", help="path file")
    # parser.add_option("--input", dest="input", default="", help="input file")
    # parser.add_option("--output", dest="output", default="", help="output file")
    # (options, args) = parser.parse_args()
    # path = options.path
    # input = options.input
    # output = options.output
    # vocab = [s for s in pickle.load(open(path, "rb"))]
    # # vocab = [s for s in pickle.load(open(path, "rb"))][:5000]
    #
    # try:
    #     GenerateCSC(infile=input, outfile=output, vocab=vocab)
    #     print("All Finished.")
    # except Exception as err:
    #     print(err)

    obj = BertMlm(bert_path="/data_local/plm_models/chinese_L-12_H-768_A-12/")

    obj.mlm_predict(["那件事令我后悔莫及，如果世界上有后悔药，那我一定[MASK]去买！",
                     "那件事令我后悔莫及，如果世界上有后悔药，那我一定回去买！",
                     "我们立刻前往土著人的村庄，发[MASK]表哥在一个笼子里啊！",
                     "我们立刻前往土著人的村庄，发生表哥在一个笼子里啊！",
                     "我们立刻前往土著人的村庄，发生表哥在一个笼子里啊！",
                     "彤红的手紧紧地抓着绳子，显然十分坚[MASK]的在行走。",
                     "彤红的手紧紧地抓着绳子，显然十分[MASK]难的在行走。",
                     "老师的汗水不住地滴下，时间也不[MASK]地流逝。",
                     "童年可真是美好，所以这件事成了我脑海里难以忘记的时刻，[MASK]，真想回到童年呀!可是时间是不能倒流的……"])

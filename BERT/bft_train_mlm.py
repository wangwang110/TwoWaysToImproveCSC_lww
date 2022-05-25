# -*- coding: UTF-8 -*-
import os
import pickle
import numpy as np
import argparse
import operator
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils.getF1 import sent_mertic, token_mertic
from utils.preprocess import is_chinese
from transformers import BertForMaskedLM, BertModel, BertConfig, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from model import BertFineTune, BertFineTuneMac, construct, BertDataset, BFTLogitGen, readAllConfusionSet


class Trainer:
    def __init__(self, bert, optimizer, scheduler, tb_writer, tokenizer, device):
        self.model = bert
        self.optim = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.tb_writer = tb_writer
        self.device = device
        self.confusion_set = readAllConfusionSet('./save/confusion.file')

        # bert的词典
        self.vob2id = {}
        self.id2vob = {}
        with open(args.bert_path + "vocab.txt", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.id2vob.setdefault(i, line.strip())
                self.vob2id.setdefault(line.strip(), i)

        # 动态掩码,构造数据
        vocab = pickle.load(open("../large_data/zh_wiki_sent/wiki_vocab.pkl", "rb"))
        self.vocab = [s for s in vocab if s in self.vob2id and is_chinese(s)]

        if args.vocab_refine:
            self.bert_vocab_refine = pickle.load(open("../large_data/bert_vocab_refine.pkl", "rb"))
            words = list(self.bert_vocab_refine.keys())  # 5628
            # 转化为分类id
            self.cls_w2id = {w: idx for idx, w in enumerate(words)}
            self.cls_w2id["KEEP"] = len(self.cls_w2id)
            self.cls_id2w = {idx: w for w, idx in self.cls_w2id.items()}
        else:
            self.cls_w2id = self.vob2id
            self.cls_id2w = self.id2vob

    def replace_token(self, lines, ratio=0.25):
        """
        # 随机选取id，查看id对应的词是否在候选集中
        # 如果在90%替换为候选词，10%随机选取token（来自哪里呢？统计字典）
        # 直到1/4的字符被替换
        :param line:
        :return:
        """
        generate_srcs = []
        for line in lines:
            num = len(line)
            tokens = list(line)
            index_li = [i for i in range(num)]
            # 有可能是引入的错误个数太多了
            up_num = num * ratio
            # 有可能是引入的错误个数太多了
            np.random.shuffle(index_li)
            count = 0
            for i in index_li:
                if tokens[i] in self.confusion_set:
                    count += 1
                    if np.random.rand(1) > 0.9:  # 这个比例是否要控制
                        idx = np.random.randint(0, len(self.vocab))
                        tokens[i] = self.vocab[idx]
                    else:
                        token_conf_set = self.confusion_set[tokens[i]]
                        idx = np.random.randint(0, len(token_conf_set))
                        tokens[i] = list(token_conf_set)[idx]
                    if count == up_num:
                        break
            generate_srcs.append("".join(tokens))
        return generate_srcs

    def train(self, train, step):
        self.model.train()
        total_loss = 0
        last_loss = 0
        for batch in train:
            step += 1
            # # 每个batch进行数据构造，类似于动态mask
            if args.dynamic_mask:
                generate_srcs = self.replace_token(batch['output'])
                batch['input'] = generate_srcs

            inputs, outputs = self.help_vectorize(batch)
            max_len = 180
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'][:, :max_len], \
                                                    inputs['token_type_ids'][:, :max_len], \
                                                    inputs['attention_mask'][:, :max_len]
            output_ids, output_token_label = outputs['input_ids'][:, :max_len], outputs['token_labels'][:, :max_len]
            outputs = self.model(input_ids, input_tyi, input_attn_mask, output_ids, output_token_label)
            # loss 在模型里面计算好
            if args.cpoloss:
                ori_c_loss = outputs[-1] + outputs[-3]
            else:
                ori_c_loss = (1 - args.multitask_weight) * outputs[-1] + args.multitask_weight * outputs[-2]

            c_loss = ori_c_loss / args.gradient_accumulation_steps
            c_loss.backward()
            total_loss += c_loss.item()
            print(c_loss.item())
            if step % args.gradient_accumulation_steps == 0:
                # print(c_loss.item())
                self.optim.step()
                self.optim.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()

            if step % args.logging_step == 0:
                # 便于在tensorboard中查看
                lr_val = args.learning_rate
                if self.scheduler is not None:
                    lr_val = scheduler.get_last_lr()[0]
                # 学习率变化
                self.tb_writer.add_scalar('lr', lr_val, step)
                # loss变化
                tb_writer.add_scalar('loss', (total_loss - last_loss) / args.gradient_accumulation_steps, step)
                last_loss = total_loss

        return total_loss, step

    def test(self, test):
        self.model.eval()
        total_loss = 0
        for batch in test:
            inputs, outputs = self.help_vectorize(batch)
            max_len = 180
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'][:, :max_len], \
                                                    inputs['token_type_ids'][:, :max_len], \
                                                    inputs['attention_mask'][:, :max_len]
            output_ids, output_token_label = outputs['input_ids'][:, :max_len], outputs['token_labels'][:, :max_len]
            outputs = self.model(input_ids, input_tyi, input_attn_mask, output_ids, output_token_label)
            c_loss = outputs[-1]
            total_loss += c_loss.item()
        return total_loss

    def save(self, name):
        if isinstance(self.model, nn.DataParallel):
            torch.save(self.model.module.state_dict(), name)
        else:
            torch.save(self.model.state_dict(), name)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    # def testSet(self, test):
    #     """
    #     :param test:
    #     :return:
    #     """
    #     self.model.eval()
    #     sen_acc = 0
    #     setsum = 0
    #     sen_mod = 0
    #     sen_mod_acc = 0
    #     sen_tar_mod = 0
    #
    #     d_sen_acc = 0
    #     d_sen_mod = 0
    #     d_sen_mod_acc = 0
    #     d_sen_tar_mod = 0
    #
    #     d_sen_acc2 = 0
    #     d_sen_mod2 = 0
    #     d_sen_mod_acc2 = 0
    #
    #     for batch in test:
    #         inputs, outputs = self.help_vectorize(batch)
    #         max_len = 180
    #         input_ids, input_tyi, input_attn_mask = inputs['input_ids'][:, :max_len], \
    #                                                 inputs['token_type_ids'][:, :max_len], \
    #                                                 inputs['attention_mask'][:, :max_len]
    #         input_lens = torch.sum(input_attn_mask, 1)
    #         output_ids, output_token_label = outputs['input_ids'][:, :max_len], outputs['token_labels'][:, :max_len]
    #
    #         output = self.model(input_ids, input_tyi, input_attn_mask, output_ids,
    #                             output_token_label)
    #         torken_prob = output[0]
    #         out_prob = output[1]
    #         out = out_prob.argmax(dim=-1)
    #
    #         mod_sen = [not out[i][:input_lens[i]].equal(input_ids[i][:input_lens[i]]) for i in range(len(out))]
    #         # 预测有错的句子
    #         acc_sen = [out[i][:input_lens[i]].equal(output_ids[i][:input_lens[i]]) for i in range(len(out))]
    #         # 修改正确的句子
    #         tar_sen = [not output_ids[i].equal(input_ids[i]) for i in range(len(output_ids))]
    #         # 实际有错的句子
    #
    #         sen_mod += sum(mod_sen)
    #         # 预测有错的句子
    #         sen_mod_acc += sum(np.multiply(np.array(mod_sen), np.array(acc_sen)))
    #         # 预测有错的句子里面，预测对了的句子
    #         sen_tar_mod += sum(tar_sen)
    #         # 实际有错的句子
    #         sen_acc += sum(acc_sen)
    #         # 预测对了句子，包括修正和不修正的
    #         setsum += output_ids.shape[0]
    #
    #         prob_2 = [[0 if torken_prob[i][j] < 0.5 else 1 for j in range(input_lens[i])] for i in range(len(out))]
    #         prob_ = [[0 if out[i][j] == input_ids[i][j] else 1 for j in range(input_lens[i])] for i in range(len(out))]
    #         label = [[0 if input_ids[i][j] == output_ids[i][j] else 1 for j in range(input_lens[i])] for i in
    #                  range(len(input_ids))]
    #
    #         d_acc_sen = [operator.eq(prob_[i], label[i]) for i in range(len(prob_))]
    #         d_acc_sen2 = [operator.eq(prob_2[i], label[i]) for i in range(len(prob_2))]
    #
    #         d_mod_sen = [0 if sum(prob_[i]) == 0 else 1 for i in range(len(prob_))]
    #         d_mod_sen2 = [0 if sum(prob_2[i]) == 0 else 1 for i in range(len(prob_2))]
    #
    #         d_tar_sen = [0 if sum(label[i]) == 0 else 1 for i in range(len(label))]
    #
    #         d_sen_mod += sum(d_mod_sen)
    #         d_sen_mod2 += sum(d_mod_sen2)
    #         # 预测有错的句子
    #         d_sen_mod_acc += sum(np.multiply(np.array(d_mod_sen), np.array(d_acc_sen)))
    #         d_sen_mod_acc2 += sum(np.multiply(np.array(d_mod_sen2), np.array(d_acc_sen2)))
    #         # 预测有错的里面，位置预测正确的
    #         d_sen_tar_mod += sum(d_tar_sen)
    #         # 实际有错的句子
    #         d_sen_acc += sum(d_acc_sen)
    #         d_sen_acc2 += sum(d_acc_sen2)
    #     #
    #     d_precision2 = d_sen_mod_acc2 / d_sen_mod2
    #     d_recall2 = d_sen_mod_acc2 / d_sen_tar_mod
    #     d_F12 = 2 * d_precision2 * d_recall2 / (d_precision2 + d_recall2)
    #     #
    #
    #     print("new detection sentence accuracy:{0},precision:{1},recall:{2},F1:{3}".format(d_sen_acc2 / setsum,
    #                                                                                        d_precision2,
    #                                                                                        d_recall2, d_F12))
    #
    #     d_precision = d_sen_mod_acc / d_sen_mod
    #     d_recall = d_sen_mod_acc / d_sen_tar_mod
    #     d_F1 = 2 * d_precision * d_recall / (d_precision + d_recall)
    #
    #     c_precision = sen_mod_acc / sen_mod
    #     c_recall = sen_mod_acc / sen_tar_mod
    #     c_F1 = 2 * c_precision * c_recall / (c_precision + c_recall)
    #
    #     print("detection sentence accuracy:{0},precision:{1},recall:{2},F1:{3}".format(d_sen_acc / setsum, d_precision,
    #                                                                                    d_recall, d_F1))
    #     print("correction sentence accuracy:{0},precision:{1},recall:{2},F1:{3}".format(sen_acc / setsum,
    #                                                                                     sen_mod_acc / sen_mod,
    #                                                                                     sen_mod_acc / sen_tar_mod,
    #                                                                                     c_F1))
    #     print("sentence target modify:{0},sentence sum:{1},sentence modified accurate:{2}".format(sen_tar_mod, setsum,
    #                                                                                               sen_mod_acc))
    #     # accuracy, precision, recall, F1
    #     return sen_acc / setsum, sen_mod_acc / sen_mod, sen_mod_acc / sen_tar_mod, c_F1

    def testSet_true(self, test):
        """
        修正之后的句子拿来对比，这才是最可靠的
        1. bert词表
        2. refine 过后的词表
        :param test:
        :return:
        """
        self.model.eval()
        all_srcs = []
        all_trgs = []
        all_pres = []
        for batch in test:
            all_srcs.extend(batch["input"])
            all_trgs.extend(batch["output"])
            inputs, outputs = self.help_vectorize(batch)
            max_len = 180
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'][:, :max_len], \
                                                    inputs['token_type_ids'][:, :max_len], \
                                                    inputs['attention_mask'][:, :max_len]

            output_ids, output_token_label = outputs['input_ids'][:, :max_len], outputs['token_labels'][:, :max_len]

            output = self.model(input_ids, input_tyi, input_attn_mask, output_ids,
                                output_token_label)
            torken_prob = output[0]
            out_prob = output[1]

            out = out_prob.argmax(dim=-1)
            num = len(batch["input"])
            if not args.vocab_refine:
                for i in range(num):
                    src = batch["input"][i]
                    tokens = list(src)
                    for j in range(len(tokens) + 1):
                        if out[i][j + 1] != input_ids[i][j + 1] and out[i][j + 1] not in [0, 100, 101, 102]:
                            val = out[i][j + 1].item()
                            if j < len(tokens):
                                tokens[j] = self.cls_id2w[val]
                    out_sent = "".join(tokens)
                    all_pres.append(out_sent)
            else:
                for i in range(num):
                    src = batch["input"][i]
                    tokens = list(src)
                    for j in range(len(tokens) + 1):
                        if out[i][j + 1] not in [self.cls_w2id["KEEP"], 0, 1, 2, 3]:
                            val = out[i][j + 1].item()
                            if j < len(tokens):
                                tokens[j] = self.cls_id2w[val]
                    out_sent = "".join(tokens)
                    all_pres.append(out_sent)

        accuracy, precision, recall, F1 = sent_mertic(all_srcs, all_pres, all_trgs)
        token_mertic(all_srcs, all_pres, all_trgs)
        # path = os.path.dirname(args.load_path) + "/test.out"
        # with open(path, "w", encoding="utf-8") as f:
        #     for src, text in zip(all_srcs, all_pres):
        #         f.write(src + " " + text + "\n")
        return accuracy, precision, recall, F1

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

    def trg2vec(self, src, trg, max_seq_length):
        """
        :param src:
        :return:
        """
        tokens_a = [a for a in src]
        tokens_b = [b for b in trg]
        input_ids = []
        segment_ids = []
        keep_id = self.cls_w2id["KEEP"]
        input_ids.append(self.cls_w2id["[CLS]"])
        segment_ids.append(0)
        for a, b in zip(tokens_a, tokens_b):
            if a == b:
                input_ids.append(keep_id)
            elif b in self.bert_vocab_refine:
                input_ids.append(self.cls_w2id[b])
            else:
                input_ids.append(self.cls_w2id["[UNK]"])
            segment_ids.append(0)
        input_ids.append(self.cls_w2id["[SEP]"])
        segment_ids.append(0)

        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return input_ids, input_mask, segment_ids

    def help_vectorize(self, batch):
        """
        :param batch:
        :return:
        """
        src_li, trg_li = batch['input'], batch['output']
        max_seq_length = max([len(src) for src in src_li]) + 2
        inputs = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
        outputs = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}

        for src, trg in zip(src_li, trg_li):
            input_ids, input_mask, segment_ids = self.text2vec(src, max_seq_length)
            inputs['input_ids'].append(input_ids)
            inputs['token_type_ids'].append(segment_ids)
            inputs['attention_mask'].append(input_mask)
            if args.vocab_refine:
                output_ids, output_mask, output_segment_ids = self.trg2vec(src, trg, max_seq_length)
            else:
                output_ids, output_mask, output_segment_ids = self.text2vec(trg, max_seq_length)
            outputs['input_ids'].append(output_ids)
            outputs['token_type_ids'].append(output_segment_ids)
            outputs['attention_mask'].append(output_mask)

        inputs['input_ids'] = torch.tensor(np.array(inputs['input_ids'])).to(self.device)
        inputs['token_type_ids'] = torch.tensor(np.array(inputs['token_type_ids'])).to(self.device)
        inputs['attention_mask'] = torch.tensor(np.array(inputs['attention_mask'])).to(self.device)

        outputs['input_ids'] = torch.tensor(np.array(outputs['input_ids'])).to(self.device)
        outputs['token_type_ids'] = torch.tensor(np.array(outputs['token_type_ids'])).to(self.device)
        outputs['attention_mask'] = torch.tensor(np.array(outputs['attention_mask'])).to(self.device)

        # 每个token是否错误二分类
        if args.vocab_refine:
            token_labels = [
                [0 if outputs['input_ids'][i][j] in [self.cls_w2id["KEEP"], 0, 1, 2, 3] else 1
                 for j in range(inputs['input_ids'].size()[1])] for i in range(len(batch['input']))]
        else:
            token_labels = [
                [0 if inputs['input_ids'][i][j] == outputs['input_ids'][i][j] else 1
                 for j in range(inputs['input_ids'].size()[1])] for i in range(len(batch['input']))]

        outputs["token_labels"] = torch.tensor(token_labels, dtype=torch.float32).to(self.device).unsqueeze(-1)
        return inputs, outputs


class Tester:
    def __init__(self, bert, tokenizer, device):
        self.model = bert
        self.device = device
        self.tokenizer = tokenizer
        self.confusion_set = readAllConfusionSet('./save/confusion.file')

        # bert的词典
        self.vob2id = {}
        self.id2vob = {}
        with open(args.bert_path + "vocab.txt", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.id2vob.setdefault(i, line.strip())
                self.vob2id.setdefault(line.strip(), i)

        # 动态掩码,构造数据
        vocab = pickle.load(open("../large_data/zh_wiki_sent/wiki_vocab.pkl", "rb"))
        self.vocab = [s for s in vocab if s in self.vob2id and is_chinese(s)]

        if args.vocab_refine:
            self.bert_vocab_refine = pickle.load(open("../large_data/bert_vocab_refine.pkl", "rb"))
            words = list(self.bert_vocab_refine.keys())  # 5628
            # 转化为分类id
            self.cls_w2id = {w: idx for idx, w in enumerate(words)}
            self.cls_w2id["KEEP"] = len(self.cls_w2id)
            self.cls_id2w = {idx: w for w, idx in self.cls_w2id.items()}
        else:
            self.cls_w2id = self.vob2id
            self.cls_id2w = self.id2vob

    def testSet(self, test):
        """
        修正之后的句子拿来对比，这才是最可靠的
        1. bert词表
        2. refine 过后的词表
        :param test:
        :return:
        """
        self.model.eval()
        all_srcs = []
        all_trgs = []
        all_pres = []
        for batch in test:
            all_srcs.extend(batch["input"])
            all_trgs.extend(batch["output"])
            inputs, outputs = self.help_vectorize(batch)
            max_len = 180
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'][:, :max_len], \
                                                    inputs['token_type_ids'][:, :max_len], \
                                                    inputs['attention_mask'][:, :max_len]

            output_ids, output_token_label = outputs['input_ids'][:, :max_len], outputs['token_labels'][:, :max_len]

            output = self.model(input_ids, input_tyi, input_attn_mask, output_ids,
                                output_token_label)
            torken_prob = output[0]
            out_prob = output[1]

            out = out_prob.argmax(dim=-1)
            num = len(batch["input"])
            if not args.vocab_refine:
                for i in range(num):
                    src = batch["input"][i]
                    tokens = list(src)
                    for j in range(len(tokens) + 1):
                        if out[i][j + 1] != input_ids[i][j + 1] and out[i][j + 1] not in [0, 100, 101, 102]:
                            val = out[i][j + 1].item()
                            if j < len(tokens):
                                tokens[j] = self.cls_id2w[val]
                    out_sent = "".join(tokens)
                    all_pres.append(out_sent)
            else:
                for i in range(num):
                    src = batch["input"][i]
                    tokens = list(src)
                    for j in range(len(tokens) + 1):
                        if out[i][j + 1] not in [self.cls_w2id["KEEP"], 0, 1, 2, 3]:
                            val = out[i][j + 1].item()
                            if j < len(tokens):
                                tokens[j] = self.cls_id2w[val]
                    out_sent = "".join(tokens)
                    all_pres.append(out_sent)

        accuracy, precision, recall, F1 = sent_mertic(all_srcs, all_pres, all_trgs)
        # token_mertic(all_srcs, all_pres, all_trgs)
        # path = os.path.dirname(args.load_path) + "/test.out"
        # with open(path, "w", encoding="utf-8") as f:
        #     for src, text in zip(all_srcs, all_pres):
        #         f.write(src + " " + text + "\n")
        return accuracy, precision, recall, F1



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

    def trg2vec(self, src, trg, max_seq_length):
        """
        :param src:
        :return:
        """
        tokens_a = [a for a in src]
        tokens_b = [b for b in trg]
        input_ids = []
        segment_ids = []
        keep_id = self.cls_w2id["KEEP"]
        input_ids.append(self.cls_w2id["[CLS]"])
        segment_ids.append(0)
        for a, b in zip(tokens_a, tokens_b):
            if a == b:
                input_ids.append(keep_id)
            elif b in self.bert_vocab_refine:
                input_ids.append(self.cls_w2id[b])
            else:
                input_ids.append(self.cls_w2id["[UNK]"])
            segment_ids.append(0)
        input_ids.append(self.cls_w2id["[SEP]"])
        segment_ids.append(0)

        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return input_ids, input_mask, segment_ids

    def help_vectorize(self, batch):
        """
        :param batch:
        :return:
        """
        src_li, trg_li = batch['input'], batch['output']
        max_seq_length = max([len(src) for src in src_li]) + 2
        inputs = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
        outputs = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}

        for src, trg in zip(src_li, trg_li):
            input_ids, input_mask, segment_ids = self.text2vec(src, max_seq_length)
            inputs['input_ids'].append(input_ids)
            inputs['token_type_ids'].append(segment_ids)
            inputs['attention_mask'].append(input_mask)
            if args.vocab_refine:
                output_ids, output_mask, output_segment_ids = self.trg2vec(src, trg, max_seq_length)
            else:
                output_ids, output_mask, output_segment_ids = self.text2vec(trg, max_seq_length)
            outputs['input_ids'].append(output_ids)
            outputs['token_type_ids'].append(output_segment_ids)
            outputs['attention_mask'].append(output_mask)

        inputs['input_ids'] = torch.tensor(np.array(inputs['input_ids'])).to(self.device)
        inputs['token_type_ids'] = torch.tensor(np.array(inputs['token_type_ids'])).to(self.device)
        inputs['attention_mask'] = torch.tensor(np.array(inputs['attention_mask'])).to(self.device)

        outputs['input_ids'] = torch.tensor(np.array(outputs['input_ids'])).to(self.device)
        outputs['token_type_ids'] = torch.tensor(np.array(outputs['token_type_ids'])).to(self.device)
        outputs['attention_mask'] = torch.tensor(np.array(outputs['attention_mask'])).to(self.device)

        # 每个token是否错误二分类
        if args.vocab_refine:
            token_labels = [
                [0 if outputs['input_ids'][i][j] in [self.cls_w2id["KEEP"], 0, 1, 2, 3] else 1
                 for j in range(inputs['input_ids'].size()[1])] for i in range(len(batch['input']))]
        else:
            token_labels = [
                [0 if inputs['input_ids'][i][j] == outputs['input_ids'][i][j] else 1
                 for j in range(inputs['input_ids'].size()[1])] for i in range(len(batch['input']))]

        outputs["token_labels"] = torch.tensor(token_labels, dtype=torch.float32).to(self.device).unsqueeze(-1)
        return inputs, outputs


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
    # np.random.seed(seed)


def str2bool(strIn):
    if strIn.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif strIn.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        print(strIn)
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == "__main__":
    import time

    # add arguments
    parser = argparse.ArgumentParser(description="choose which model")
    parser.add_argument('--task_name', type=str, default='bert_pretrain')
    parser.add_argument('--gpu_num', type=int, default=2)
    parser.add_argument('--load_model', type=str2bool, nargs='?', const=False)
    parser.add_argument('--load_path', type=str, default='./save/13_train_seed0_1.pkl')
    parser.add_argument('--do_train', type=str2bool, nargs='?', const=False)
    parser.add_argument('--train_data', type=str, default='../data/13train.txt')
    parser.add_argument('--do_valid', type=str2bool, nargs='?', const=False)
    parser.add_argument('--valid_data', type=str, default='../data/13valid.txt')
    parser.add_argument('--do_test', type=str2bool, nargs='?', const=False)
    parser.add_argument('--test_data', type=str, default='../data/13test.txt')
    parser.add_argument('--batch_size', type=int, default=20)

    # gradient_accumulation_steps
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",
                        )
    # logging_setp
    parser.add_argument("--logging_step", type=int, default=10,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",
                        )
    # 预训练模型路径
    parser.add_argument('--bert_path', type=str, default="/data_local/plm_models/chinese_L-12_H-768_A-12/")
    # 日志路径
    parser.add_argument('--log_path', type=str, default='./doc_conj_train/log/')
    # 是否使用warmup
    parser.add_argument("--do_warmup", action='store_true', help="warmup")
    parser.add_argument("--warmup_step", type=int, default=1000)
    # 是否复用bert预测层参数
    parser.add_argument("--mlm", action='store_true', help="bert_mlm")
    # 是否使用动态mask
    parser.add_argument("--dynamic_mask", action='store_true', help="dynamic_mask")
    # 是否进行词汇过滤
    parser.add_argument("--vocab_refine", action='store_true', help="vocab_refine")
    # 多任务训练权值，token二分类
    parser.add_argument("--multitask_weight", type=float, default=0.7)

    # 错误汉字预测loss，权重
    parser.add_argument("--error_weight", type=float, default=1.0)

    # 是否加入cpoloss
    parser.add_argument("--cpoloss", action='store_true', help="cpoloss")

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--do_save', type=str2bool, nargs='?', const=False)
    parser.add_argument('--save_dir', type=str, default='../save')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    # 参数控制代码流程
    # 实验的可复现性和一致性

    task_name = args.task_name
    # print("----python script: " + os.path.basename(__file__) + "----")
    # print("----Task: " + task_name + " begin !----")
    # print("----Model base: " + args.load_path + "----")
    # print("----Train data: " + args.train_data + "----")
    # print("----Batch size: " + str(args.batch_size) + "----")
    # print(args)

    setup_seed(int(args.seed))

    tb_writer = SummaryWriter(args.log_path)
    start = time.time()

    device_ids = [i for i in range(int(args.gpu_num))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bert_path = "/data_local/plm_models/chinese_L-12_H-768_A-12/"
    if args.mlm:
        # 复用bert分类层参数
        bert = BertForMaskedLM.from_pretrained(args.bert_path, return_dict=True)
    else:
        bert = BertModel.from_pretrained(args.bert_path, return_dict=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    config = BertConfig.from_pretrained(args.bert_path)

    if args.mlm:
        model = BertFineTuneMac(bert, tokenizer, device, device_ids, args).to(device)
    else:
        model = BertFineTune(bert, tokenizer, device, device_ids, args).to(device)

    if args.load_model:
        model.load_state_dict(torch.load(args.load_path))

    model = nn.DataParallel(model, device_ids)

    if args.do_train:
        train = construct(args.train_data)
        train = BertDataset(tokenizer, train)
        train = DataLoader(train, batch_size=int(args.batch_size), shuffle=True)

    if args.do_valid:
        valid = construct(args.valid_data)
        valid = BertDataset(tokenizer, valid)
        valid = DataLoader(valid, batch_size=int(args.batch_size), shuffle=True)

    if args.do_test:
        test = construct(args.test_data)
        test = BertDataset(tokenizer, test)
        test = DataLoader(test, batch_size=int(args.batch_size), shuffle=False)

    if args.do_train:
        # 总共的更新次数
        t_total = len(train) // args.gradient_accumulation_steps * int(args.epoch)
        # print("update t_total:{}".format(t_total))

        if args.do_warmup:
            optimizer = AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=0.01)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.5 * t_total,
                                                        num_training_steps=t_total)
        else:
            optimizer = Adam(model.parameters(), float(args.learning_rate))
            scheduler = None

        trainer = Trainer(model, optimizer, scheduler, tb_writer, tokenizer, device)
        max_f1 = 0
        best_epoch = 0
        step = 0  # 更新次数
        for e in range(int(args.epoch)):
            train_loss, step = trainer.train(train, step)
            if args.do_valid:
                pass
                valid_loss = trainer.test(valid)
                valid_acc, valid_pre, valid_rec, valid_f1 = trainer.testSet_true(valid)
                print(task_name, ",epoch {0},train_loss: {1},valid_loss: {2}".format(e + 1, train_loss, valid_loss))

                # don't have to save model
                if valid_f1 <= max_f1:
                    print("Time cost:", time.time() - start, "s")
                    print("-" * 10)
                    continue
                max_f1 = valid_f1
            else:
                print(task_name, ",epoch {0},train_loss:{1}".format(e + 1, train_loss))

            best_epoch = e + 1
            if args.do_save:
                model_save_path = args.save_dir + '/epoch{0}.pkl'.format(e + 1)
                trainer.save(model_save_path)
                # print("save model done!")
            # print("Time cost:", time.time() - start, "s")
            # print("-" * 10)

        model_best_path = args.save_dir + '/epoch{0}.pkl'.format(best_epoch)
        model_save_path = args.save_dir + '/model.pkl'

        # copy the best model to standard name
        os.system('cp ' + model_best_path + " " + model_save_path)
        tb_writer.close()

    if args.do_test:
        # trainer.testSet(test)
        tester = Tester(model, tokenizer, device)
        tester.testSet_true(test)

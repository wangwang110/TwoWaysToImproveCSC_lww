# -*- coding: UTF-8 -*-

import sys

sys.path.append("..")
import torch.nn as nn
import torch
import numpy as np
import argparse
from transformers import BertModel, BertConfig, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import operator
from model import BertFineTune, construct, construct_ner, BertDataset, BFTLogitGen, readAllConfusionSet
import os

"""
sighan15上的结果不一致

"""


class Trainer:
    def __init__(self, bert, optimizer, tokenizer, device):
        self.model = bert
        self.optim = optimizer
        self.tokenizer = tokenizer
        self.criterion_c = nn.NLLLoss()
        self.criterion_cls = nn.BCELoss()
        self.criterion_token_cls = nn.BCELoss(reduction="none")

        # ignore_index=0
        self.device = device
        self.confusion_set = readAllConfusionSet('/data_local/TwoWaysToImproveCSC/BERT/save/confusion.file')
        # /data_local/TwoWaysToImproveCSC/BERT/save/confusion.file
        # ../save/confusion.file

    def train(self, train):
        self.model.train()
        total_loss = 0
        for batch in train:
            # inputs = self.tokenizer(batch['input'], padding=True, truncation=True, return_tensors="pt").to(self.device)
            # outputs = self.tokenizer(batch['output'], padding=True, truncation=True, return_tensors="pt").to(
            #     self.device)

            # inputs, outputs = self.help_vectorize_ori(batch)
            # 修改分词方式
            # 排查评价方式
            inputs, outputs = self.help_vectorize(batch)
            # if inputs['input_ids'].size() != inputs_s['input_ids'].size():
            #     print("error!!!!!!!")
            #     print(batch)

            max_len = 180
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'][:, :max_len], \
                                                    inputs['token_type_ids'][:, :max_len], \
                                                    inputs['attention_mask'][:, :max_len]
            output_ids, output_tyi, output_attn_mask = outputs['input_ids'][:, :max_len], \
                                                       outputs['token_type_ids'][:, :max_len], \
                                                       outputs['attention_mask'][:, :max_len]
            output_token_label = outputs["token_labels"][:, :max_len, :]
            # batch, seq_len = input_ids.size()

            out, sent_prob, torken_prob = self.model(input_ids, input_tyi,
                                                     input_attn_mask)  # out:[batch_size,seq_len,vocab_size]

            # 实现ner多任务训练
            outputs_ner = []
            batch_size, seq_len = input_ids.size()
            for s in batch['output_ner']:
                init_li = [0.0] * seq_len
                real_li = s.strip().split(" ")[:max_len-1]
                for i in range(len(real_li)):
                    if real_li[i] == "1":
                        init_li[i + 1] = 1.0
                outputs_ner.append(init_li)
            outputs_ner = torch.tensor(outputs_ner).to(self.device)

            ori_c_loss = self.criterion_c(out.transpose(1, 2), output_ids)
            sent_loss = self.criterion_cls(sent_prob, outputs["labels"])

            seq_loss = self.criterion_token_cls(torken_prob.reshape(batch_size * seq_len, 1),
                                                outputs_ner.reshape(batch_size * seq_len, 1))

            # seq_loss = self.criterion_token_cls(torken_prob.view(-1, 1),
            #                                     output_token_label.view(-1, 1))

            seq_loss = seq_loss.view(input_ids.size()[0], input_ids.size()[1]) * input_attn_mask
            seq_loss = torch.sum(seq_loss) / torch.sum(input_attn_mask)

            # c_loss = ori_c_loss
            c_loss = ori_c_loss + seq_loss

            # c_loss = 0.5*ori_c_loss + 0.5*seq_loss

            total_loss += c_loss.item()
            # total_loss += sent_loss.item()
            # total_loss += seq_loss.item()
            print(c_loss.item())
            self.optim.zero_grad()
            c_loss.backward()
            # backward始终用的这一个
            self.optim.step()
        return total_loss

    def test(self, test):
        self.model.eval()
        total_loss = 0
        for batch in test:
            # inputs = self.tokenizer(batch['input'], padding=True, truncation=True, return_tensors="pt").to(self.device)
            # outputs = self.tokenizer(batch['output'], padding=True, truncation=True, return_tensors="pt").to(
            #     self.device)

            inputs, outputs = self.help_vectorize_ori(batch)
            max_len = 180
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'][:, :max_len], \
                                                    inputs['token_type_ids'][:, :max_len], \
                                                    inputs['attention_mask'][:, :max_len]
            output_ids, output_tyi, output_attn_mask = outputs['input_ids'][:, :max_len], \
                                                       outputs['token_type_ids'][:, :max_len], \
                                                       outputs['attention_mask'][:, :max_len]

            out, _, _ = self.model(input_ids, input_tyi, input_attn_mask)
            output_token_label = outputs["token_labels"][:, :max_len, :]
            # batch, seq_len = input_ids.size()

            out, sent_prob, torken_prob = self.model(input_ids, input_tyi,
                                                     input_attn_mask)  # out:[batch_size,seq_len,vocab_size]

            # 实现ner多任务训练
            outputs_ner = []
            batch_size, seq_len = input_ids.size()
            for s in batch['output_ner']:
                init_li = [0.0] * seq_len
                real_li = s.strip().split(" ")[:max_len - 1]
                for i in range(len(real_li)):
                    if real_li[i] == "1":
                        init_li[i + 1] = 1.0
                outputs_ner.append(init_li)
            outputs_ner = torch.tensor(outputs_ner).to(self.device)

            ori_c_loss = self.criterion_c(out.transpose(1, 2), output_ids)
            sent_loss = self.criterion_cls(sent_prob, outputs["labels"])

            seq_loss = self.criterion_token_cls(torken_prob.reshape(batch_size * seq_len, 1),
                                                outputs_ner.reshape(batch_size * seq_len, 1))

            # seq_loss = self.criterion_token_cls(torken_prob.view(-1, 1),
            #                                     output_token_label.view(-1, 1))

            seq_loss = seq_loss.view(input_ids.size()[0], input_ids.size()[1]) * input_attn_mask
            seq_loss = torch.sum(seq_loss) / torch.sum(input_attn_mask)

            # c_loss = ori_c_loss
            c_loss = ori_c_loss + seq_loss
            total_loss += c_loss.item()
        return total_loss

    def save(self, name):
        if isinstance(self.model, nn.DataParallel):
            torch.save(self.model.module.state_dict(), name)
        else:
            torch.save(self.model.state_dict(), name)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def testSet(self, test):
        self.model.eval()
        sen_acc = 0
        setsum = 0
        sen_mod = 0
        sen_mod_acc = 0
        sen_tar_mod = 0
        d_sen_acc = 0
        d_sen_mod = 0
        d_sen_mod_acc = 0
        d_sen_tar_mod = 0
        for batch in test:
            # inputs = self.tokenizer(batch['input'], padding=True, truncation=True, return_tensors="pt").to(self.device)
            # outputs = self.tokenizer(batch['output'], padding=True, truncation=True, return_tensors="pt").to(
            #     self.device)
            inputs, outputs = self.help_vectorize_ori(batch)
            max_len = 180
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'][:, :max_len], \
                                                    inputs['token_type_ids'][:, :max_len], \
                                                    inputs['attention_mask'][:, :max_len]
            input_lens = torch.sum(input_attn_mask, 1)
            output_ids, output_tyi, output_attn_mask = outputs['input_ids'][:, :max_len], \
                                                       outputs['token_type_ids'][:, :max_len], \
                                                       outputs['attention_mask'][:, :max_len]
            out, _, _ = self.model(input_ids, input_tyi, input_attn_mask)
            out = out.argmax(dim=-1)
            mod_sen = [not out[i][:input_lens[i]].equal(input_ids[i][:input_lens[i]]) for i in range(len(out))]
            # 预测有错的句子
            acc_sen = [out[i][:input_lens[i]].equal(output_ids[i][:input_lens[i]]) for i in range(len(out))]
            # 修改正确的句子
            tar_sen = [not output_ids[i].equal(input_ids[i]) for i in range(len(output_ids))]
            # 实际有错的句子

            sen_mod += sum(mod_sen)
            # 预测有错的句子
            sen_mod_acc += sum(np.multiply(np.array(mod_sen), np.array(acc_sen)))
            # 预测有错的句子里面，预测对了的句子
            sen_tar_mod += sum(tar_sen)
            # 实际有错的句子
            # sen_acc += sum([out[i].equal(output_ids[i]) for i in range(len(out))])
            sen_acc += sum(acc_sen)
            # 预测对了句子，包括修正和不修正的

            #
            setsum += output_ids.shape[0]

            prob_ = [[0 if out[i][j] == input_ids[i][j] else 1 for j in range(input_lens[i])] for i in range(len(out))]
            label = [[0 if input_ids[i][j] == output_ids[i][j] else 1 for j in
                      range(input_lens[i])] for i in range(len(input_ids))]
            d_acc_sen = [operator.eq(prob_[i], label[i]) for i in range(len(prob_))]
            d_mod_sen = [0 if sum(prob_[i]) == 0 else 1 for i in range(len(prob_))]
            d_tar_sen = [0 if sum(label[i]) == 0 else 1 for i in range(len(label))]
            d_sen_mod += sum(d_mod_sen)
            # 预测有错的句子
            d_sen_mod_acc += sum(np.multiply(np.array(d_mod_sen), np.array(d_acc_sen)))
            # 预测有错的里面，位置预测正确的
            d_sen_tar_mod += sum(d_tar_sen)
            # 实际有错的句子
            d_sen_acc += sum(d_acc_sen)
        d_precision = d_sen_mod_acc / d_sen_mod
        d_recall = d_sen_mod_acc / d_sen_tar_mod
        d_F1 = 2 * d_precision * d_recall / (d_precision + d_recall)
        c_precision = sen_mod_acc / sen_mod
        c_recall = sen_mod_acc / sen_tar_mod
        c_F1 = 2 * c_precision * c_recall / (c_precision + c_recall)
        print("detection sentence accuracy:{0},precision:{1},recall:{2},F1:{3}".format(d_sen_acc / setsum, d_precision,
                                                                                       d_recall, d_F1))
        print("correction sentence accuracy:{0},precision:{1},recall:{2},F1:{3}".format(sen_acc / setsum,
                                                                                        sen_mod_acc / sen_mod,
                                                                                        sen_mod_acc / sen_tar_mod,
                                                                                        c_F1))
        print("sentence target modify:{0},sentence sum:{1},sentence modified accurate:{2}".format(sen_tar_mod, setsum,
                                                                                                  sen_mod_acc))
        # accuracy, precision, recall, F1
        return sen_acc / setsum, sen_mod_acc / sen_mod, sen_mod_acc / sen_tar_mod, c_F1

    def help_vectorize_ori(self, batch):
        """
        也是文本分batch，之后再向量化
        :param text_a:
        :param text_b:
        :return:
        """
        inputs = self.tokenizer(batch['input'], padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.tokenizer(batch['output'], padding=True, truncation=True, return_tensors="pt").to(
            self.device)

        # input_lens = torch.sum(inputs["attention_mask"], 1)
        # 整体的分类标签
        sent_labels = [0 if batch['input'][i] == batch['output'][i] else 1 for i in range(len(batch['input']))]
        outputs_labels = torch.tensor(sent_labels, dtype=torch.float32).to(self.device).unsqueeze(-1)
        outputs["labels"] = outputs_labels

        # 每个位置对应的分类，其中padding部分需要去掉
        token_labels = [
            [0 if inputs['input_ids'][i][j] == outputs['input_ids'][i][j] else 1
             for j in range(inputs['input_ids'].size()[1])] for i in range(len(batch['input']))]

        outputs["token_labels"] = torch.tensor(token_labels, dtype=torch.float32).to(self.device).unsqueeze(-1)
        #
        return inputs, outputs

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

        # input_lens = torch.sum(inputs["attention_mask"], 1)
        # 整体的分类标签
        sent_labels = [0 if batch['input'][i] == batch['output'][i] else 1 for i in range(len(batch['input']))]
        outputs_labels = torch.tensor(sent_labels, dtype=torch.float32).to(self.device).unsqueeze(-1)
        outputs["labels"] = outputs_labels

        # 每个位置对应的分类，其中padding部分需要去掉
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

    np.random.seed(seed)


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
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--do_save', type=str2bool, nargs='?', const=False)
    parser.add_argument('--save_dir', type=str, default='../save')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    task_name = args.task_name
    print("----Task: " + task_name + " begin !----")
    print("----Model base: " + args.load_path + "----")

    setup_seed(int(args.seed))
    start = time.time()

    # device_ids=[0,1]
    device_ids = [i for i in range(int(args.gpu_num))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    bert_path = "/data_local/plm_models/chinese_L-12_H-768_A-12/"
    bert = BertModel.from_pretrained(bert_path, return_dict=True)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    config = BertConfig.from_pretrained(bert_path)

    model = BertFineTune(bert, tokenizer, device, device_ids, is_correct_sent=True).to(device)

    if args.load_model:
        # model_dict = model.state_dict()
        # pretrained_dict = torch.load(args.load_path)
        # #
        # not_in_pretrain_model = [k for k in model_dict if k not in pretrained_dict]
        # print(not_in_pretrain_model)
        # #
        #
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
        # # 加载部分参数
        model.load_state_dict(torch.load(args.load_path))

    model = nn.DataParallel(model, device_ids)

    if args.do_train:
        train = construct_ner(args.train_data)
        train = BertDataset(tokenizer, train)
        train = DataLoader(train, batch_size=int(args.batch_size), shuffle=True)

    if args.do_valid:
        valid = construct(args.valid_data)
        valid = BertDataset(tokenizer, valid)
        valid = DataLoader(valid, batch_size=int(args.batch_size), shuffle=True)

    if args.do_test:
        test = construct(args.test_data)
        test = BertDataset(tokenizer, test)
        test = DataLoader(test, batch_size=int(args.batch_size), shuffle=True)

    optimizer = Adam(model.parameters(), float(args.learning_rate))
    # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

    trainer = Trainer(model, optimizer, tokenizer, device)
    max_f1 = 0
    best_epoch = 0

    if args.do_train:
        for e in range(int(args.epoch)):
            train_loss = trainer.train(train)

            if args.do_valid:
                valid_loss = trainer.test(valid)
                valid_acc, valid_pre, valid_rec, valid_f1 = trainer.testSet(valid)
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
                print("save model done!")
            print("Time cost:", time.time() - start, "s")
            print("-" * 10)

        model_best_path = args.save_dir + '/epoch{0}.pkl'.format(best_epoch)
        model_save_path = args.save_dir + '/model.pkl'

        # copy the best model to standard name
        os.system('cp ' + model_best_path + " " + model_save_path)

    if args.do_test:
        trainer.testSet(test)

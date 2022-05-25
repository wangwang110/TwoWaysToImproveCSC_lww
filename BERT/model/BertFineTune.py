import sys
import torch.nn as nn
import torch
import numpy as np
from transformers import BertModel, BertConfig, BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from model.dataset import BertDataset, construct
import pickle
import copy
from loss.focalloss import FocalLoss
from loss.cpoloss import CpoLoss


class BertFineTuneCpo(nn.Module):
    def __init__(self, bert, tokenizer, device, device_ids, is_correct_sent=False):
        super(BertFineTuneCpo, self).__init__()
        self.device = device
        self.config = bert.config
        self.bert = bert.to(device)

    def forward(self, input_ids, input_tyi, input_attn_mask, text_labels, det_labels):
        copy_text_labels = copy.deepcopy(text_labels)
        if text_labels is not None:
            text_labels[text_labels == 0] = -100
            # 所以会做忽略处理
        else:
            text_labels = None
        bert_outputs = self.bert(input_ids=input_ids, token_type_ids=input_tyi, attention_mask=input_attn_mask,
                                 labels=text_labels, return_dict=True, output_hidden_states=True)

        # prob = self.detection(bert_outputs.hidden_states[-1])
        if text_labels is None:
            outputs = (0, bert_outputs.logits)
        else:
            cpo_loss_fct = CpoLoss()
            # pad部分不计算损失
            # active_loss = input_attn_mask.view(-1, prob.shape[1]) == 1
            # active_probs = prob.view(-1, prob.shape[1])[active_loss]
            # active_labels = det_labels[active_loss]
            cpo_loss = cpo_loss_fct(bert_outputs.logits, copy_text_labels, input_attn_mask)

            # 检错loss，纠错loss，检错输出，纠错输出
            outputs = (cpo_loss,
                       bert_outputs.loss,
                       bert_outputs.logits)

        return outputs


class BertFineTuneMac(nn.Module):
    def __init__(self, bert, tokenizer, device, device_ids, is_correct_sent=False):
        super(BertFineTuneMac, self).__init__()
        self.device = device
        self.config = bert.config
        self.bert = bert.to(device)

        self.is_correct_sent = is_correct_sent

        if self.is_correct_sent:
            hidden_size = self.config.to_dict()['hidden_size']
            self.detection = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid().to(device)
            # 每个位置进行分类
            # self.dense = nn.Linear(hidden_size, hidden_size)
            # self.activation = nn.Tanh()

    def forward(self, input_ids, input_tyi, input_attn_mask, text_labels=None, det_labels=None, ignore=False):
        if text_labels is not None:
            text_labels[text_labels == 0] = -100
            if ignore:
                text_labels[text_labels == 101] = -100
                text_labels[text_labels == 102] = -100

        else:
            text_labels = None
        bert_outputs = self.bert(input_ids=input_ids, token_type_ids=input_tyi, attention_mask=input_attn_mask,
                                 labels=text_labels, return_dict=True, output_hidden_states=True)

        prob = self.detection(bert_outputs.hidden_states[-1])
        if text_labels is None:
            outputs = (prob, bert_outputs.logits)
        else:
            det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid')
            # pad部分不计算损失
            active_loss = input_attn_mask.view(-1, prob.shape[1]) == 1
            active_probs = prob.view(-1, prob.shape[1])[active_loss]
            active_labels = det_labels[active_loss]
            det_loss = det_loss_fct(active_probs, active_labels.float())  # 检测loss（0.7）和纠正loss（0.3）
            # 检错loss，纠错loss，检错输出，纠错输出
            outputs = (det_loss,
                       bert_outputs.loss,
                       self.sigmoid(prob).squeeze(-1),
                       bert_outputs.logits)

        return outputs


# class BertFineTuneMac(nn.Module):
#     def __init__(self, bert, tokenizer, device, device_ids, args):
#         super(BertFineTuneMac, self).__init__()
#         self.device = device
#         self.config = bert.config
#         self.bert = bert.to(device)
#         self.args = args
#
#         if self.args.is_multitask:
#             hidden_size = self.config.to_dict()['hidden_size']
#             self.detection = nn.Linear(hidden_size, 1)
#             self.sigmoid = nn.Sigmoid().to(device)
#
#     def forward(self, input_ids, input_tyi, input_attn_mask, text_labels, det_labels):
#         copy_text_labels = copy.deepcopy(text_labels)
#         if text_labels is not None:
#             text_labels[text_labels == 0] = -100
#         else:
#             text_labels = None
#         bert_outputs = self.bert(input_ids=input_ids, token_type_ids=input_tyi, attention_mask=input_attn_mask,
#                                  labels=text_labels, return_dict=True, output_hidden_states=True)
#
#         prob = self.detection(bert_outputs.hidden_states[-1])
#         if text_labels is None:
#             outputs = (
#                 self.sigmoid(prob).squeeze(-1),
#                 bert_outputs.logits,)
#         else:
#             det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid')
#             # pad部分不计算损失
#             active_mask = input_attn_mask.view(-1, prob.shape[1]) == 1
#             active_probs = prob.view(-1, prob.shape[1])[active_mask]
#             active_labels = det_labels[active_mask]
#             det_loss = det_loss_fct(active_probs, active_labels.float())  # 检测loss（0.7）和纠正loss（0.3）
#             if self.args.cpoloss:
#                 cpo_loss_fct = CpoLoss()
#                 cpo_loss = cpo_loss_fct(bert_outputs.logits, copy_text_labels, input_attn_mask)
#                 outputs = (
#                     self.sigmoid(prob).squeeze(-1),
#                     bert_outputs.logits,
#                     det_loss,
#                     cpo_loss)
#             else:
#                 outputs = (
#                     self.sigmoid(prob).squeeze(-1),
#                     bert_outputs.logits,
#                     det_loss,
#                     bert_outputs.loss)
#         return outputs


class BertFineTune(nn.Module):
    def __init__(self, bert, tokenizer, device, device_ids, args):
        super(BertFineTune, self).__init__()
        self.criterion_c = nn.CrossEntropyLoss()

        self.device = device
        self.config = bert.config
        embedding_size = self.config.to_dict()['hidden_size']
        self.bert = bert.to(device)

        if self.args.is_multitask:
            hidden_size = self.config.to_dict()['hidden_size']
            self.detection = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid().to(device)

        bert_embedding = bert.embeddings
        word_embeddings_weight = bert.embeddings.word_embeddings.weight  # bert训练好的embeding table
        embeddings = nn.Parameter(word_embeddings_weight, True)  # 参数化
        bert_embedding.word_embeddings = nn.Embedding(self.config.vocab_size,
                                                      embedding_size,
                                                      _weight=embeddings)
        # 原始的bert_embedding.word_embeddings,微调的时候不调整吗

        self.linear = nn.Linear(embedding_size, self.config.vocab_size)
        self.linear.weight = embeddings  # 不是共享，只是用于初始化

    def forward(self, input_ids, input_tyi, input_attn_mask, text_labels, det_labels):
        """
        :param input_ids:
        :param input_tyi:
        :param input_attn_mask:
        :param text_labels:
        :param det_labels:
        :return:
        """
        h = self.bert(input_ids=input_ids, token_type_ids=input_tyi, attention_mask=input_attn_mask)
        out = self.linear(h.last_hidden_state)
        copy_text_labels = copy.deepcopy(text_labels)
        if text_labels is not None:
            text_labels[text_labels == 0] = -100
            loss = self.criterion_c(out, text_labels)
        else:
            text_labels = None

        det_out = self.detection(h.last_hidden_state)
        det_prob = self.sigmoid(det_out).squeeze(-1)

        if text_labels is None:
            outputs = (det_prob, out)
        else:
            det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid')
            # pad部分不计算损失
            active_mask = input_attn_mask.view(-1, det_out.shape[1]) == 1
            active_probs = det_out.view(-1, det_out.shape[1])[active_mask]
            active_labels = det_labels[active_mask]
            det_loss = det_loss_fct(active_probs, active_labels.float())  # 检测loss（0.7）和纠正loss（0.3）

            if self.args.cpoloss:
                cpo_loss_fct = CpoLoss()
                cpo_loss = cpo_loss_fct(out, copy_text_labels, input_attn_mask)
                outputs = (det_prob, out, det_loss, loss, cpo_loss)
            else:
                outputs = (det_prob, out, det_loss, loss)

        return outputs


class BertFineKeep(nn.Module):
    def __init__(self, bert, tokenizer, device, device_ids):
        super(BertFineKeep, self).__init__()
        self.device = device
        self.config = bert.config
        embedding_size = self.config.to_dict()['hidden_size']
        self.bert = bert.to(device)

        bert_vocab_refine = pickle.load(open("../large_data/bert_vocab_refine.pkl", "rb"))
        self.cls_size = len(bert_vocab_refine) + 1
        # keep

        bert_embedding = bert.embeddings
        word_embeddings_weight = bert.embeddings.word_embeddings.weight  # bert训练好的embeding table
        embeddings = nn.Parameter(word_embeddings_weight, True)  # 参数化
        bert_embedding.word_embeddings = nn.Embedding(self.config.vocab_size, embedding_size,
                                                      _weight=embeddings)  # 为什么要做这一步
        # 原始的bert_embedding.word_embeddings,微调的时候不调整吗

        word_ids = [bert_vocab_refine[key] for key in bert_vocab_refine]
        index = torch.tensor(word_ids, device="cuda")
        cls_weight = torch.index_select(word_embeddings_weight, 0, index)  # 用原来的word_embeding初始化

        add_one_weight = word_embeddings_weight[102].view(1, -1)

        # w = torch.empty(1, embedding_size)
        # add_one_weight = nn.init.normal_(w, mean=0.0, std=1.0)

        all_cls_weight = torch.cat([cls_weight, add_one_weight], 0)
        all_cls_weight = nn.Parameter(all_cls_weight, True)

        # 另外再加一个维度

        self.linear = nn.Linear(embedding_size, self.cls_size)
        self.linear.weight = all_cls_weight  # 所以不是共享，只是用于初始化

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, input_tyi, input_attn_mask):
        h = self.bert(input_ids=input_ids, token_type_ids=input_tyi, attention_mask=input_attn_mask)
        out = self.softmax(self.linear(h.last_hidden_state))  # 对数化的概率
        return out


class BertCSC(nn.Module):
    def __init__(self, bert, tokenizer, device, is_correct_sent=False):
        super(BertCSC, self).__init__()
        self.device = device
        self.config = bert.config
        embedding_size = self.config.to_dict()['hidden_size']
        self.bert = bert.to(device)

        self.is_correct_sent = is_correct_sent

        if self.is_correct_sent:
            hidden_size = self.config.to_dict()['hidden_size']
            self.cls_w = nn.Linear(self.config.to_dict()['hidden_size'], 1)
            # 每个位置进行分类
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.activation = nn.Tanh()

        self.sigmoid = nn.Sigmoid().to(device)  # 二分类吗

        bert_embedding = bert.embeddings
        word_embeddings_weight = bert.embeddings.word_embeddings.weight  # bert训练好的embeding table
        embeddings = nn.Parameter(word_embeddings_weight, True)  # 参数化
        bert_embedding.word_embeddings = nn.Embedding(self.config.vocab_size,
                                                      embedding_size, _weight=embeddings)  # 重新保存
        self.linear = nn.Linear(embedding_size, self.config.vocab_size)
        self.linear.weight = embeddings  # 所以不是共享，只是用于初始化

        self.softmax = nn.LogSoftmax(dim=-1)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, input_tyi, input_attn_mask):
        h = self.bert(input_ids=input_ids, token_type_ids=input_tyi, attention_mask=input_attn_mask)
        out = self.softmax(self.linear(h.last_hidden_state))
        if self.is_correct_sent:
            sent_cls_out = self.sigmoid(self.cls_w(h.pooler_output))
            seq_pooler_output = self.activation(self.dense(h.last_hidden_state))
            # seq_cls_out = self.sigmoid(self.cls_w(seq_pooler_output))
            seq_cls_out = self.cls_w(seq_pooler_output)
            return out, sent_cls_out, seq_cls_out
        return out


class BertPinyin(nn.Module):
    def __init__(self, bert, tokenizer, device, device_ids, num_pinyin=430, use_pinyin=True):
        super(BertPinyin, self).__init__()
        self.device = device
        self.config = bert.config
        embedding_size = self.config.to_dict()['hidden_size']
        self.bert = bert.to(device)
        self.sigmoid = nn.Sigmoid().to(device)

        self.use_pinyin = use_pinyin

        if self.use_pinyin:
            pinyin_hidden_size = 32
            self.dense_p = nn.Linear(embedding_size, pinyin_hidden_size)
            self.activation_p = nn.Tanh()
            self.pinyin_linear = nn.Linear(pinyin_hidden_size, num_pinyin)

        bert_embedding = bert.embeddings
        word_embeddings_weight = bert.embeddings.word_embeddings.weight
        embeddings = nn.Parameter(word_embeddings_weight, True)
        bert_embedding.word_embeddings = nn.Embedding(self.config.vocab_size, embedding_size, _weight=embeddings)
        self.linear = nn.Linear(embedding_size, self.config.vocab_size)
        self.linear.weight = embeddings

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, input_tyi, input_attn_mask):
        h = self.bert(input_ids=input_ids, token_type_ids=input_tyi, attention_mask=input_attn_mask)
        out = self.softmax(self.linear(h.last_hidden_state))
        if self.use_pinyin:
            pinyin_hidden = self.activation_p(self.dense_p(h.last_hidden_state))
            pinyin_input_out = self.softmax(self.pinyin_linear(pinyin_hidden))
            # 没有输出拼音，但是直接预测拼音
            return out, pinyin_input_out
        return out

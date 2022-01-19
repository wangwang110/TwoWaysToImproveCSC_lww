import sys

sys.path.append("..")
import torch.nn as nn
import torch
import numpy as np
from transformers import BertModel, BertConfig, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from model.dataset import BertDataset, construct


class BertFineTune(nn.Module):
    def __init__(self, bert, tokenizer, device, device_ids, is_correct_sent=False):
        super(BertFineTune, self).__init__()
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
                                                      embedding_size,
                                                      _weight=embeddings)  # 重新保存
        self.linear = nn.Linear(embedding_size, self.config.vocab_size)
        self.linear.weight = embeddings  # 所以不是共享，只是用于初始化

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, input_tyi, input_attn_mask):
        h = self.bert(input_ids=input_ids, token_type_ids=input_tyi, attention_mask=input_attn_mask)
        out = self.softmax(self.linear(h.last_hidden_state))
        if self.is_correct_sent:
            sent_cls_out = self.sigmoid(self.cls_w(h.pooler_output))
            seq_pooler_output = self.activation(self.dense(h.last_hidden_state))
            seq_cls_out = self.sigmoid(self.cls_w(seq_pooler_output))
            return out, sent_cls_out, seq_cls_out
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
                                                      embedding_size,
                                                      _weight=embeddings)  # 重新保存
        self.linear = nn.Linear(embedding_size, self.config.vocab_size)
        self.linear.weight = embeddings  # 所以不是共享，只是用于初始化

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, input_tyi, input_attn_mask):
        h = self.bert(input_ids=input_ids, token_type_ids=input_tyi, attention_mask=input_attn_mask)
        out = self.softmax(self.linear(h.last_hidden_state))
        if self.is_correct_sent:
            sent_cls_out = self.sigmoid(self.cls_w(h.pooler_output))
            seq_pooler_output = self.activation(self.dense(h.last_hidden_state))
            seq_cls_out = self.sigmoid(self.cls_w(seq_pooler_output))
            return out, sent_cls_out, seq_cls_out
        return out

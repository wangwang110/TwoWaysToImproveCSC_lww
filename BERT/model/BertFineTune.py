import sys

sys.path.append("..")
import torch.nn as nn
import torch
import numpy as np
from transformers import BertModel, BertConfig, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from model.dataset import BertDataset, construct
import pickle


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

        # self.sigmoid = nn.Sigmoid().to(device)  # 二分类吗

        bert_embedding = bert.embeddings
        word_embeddings_weight = bert.embeddings.word_embeddings.weight  # bert训练好的embeding table
        embeddings = nn.Parameter(word_embeddings_weight, True)  # 参数化
        bert_embedding.word_embeddings = nn.Embedding(self.config.vocab_size,
                                                      embedding_size,
                                                      _weight=embeddings)  # 为什么要做这一步
        # 原始的bert_embedding.word_embeddings,微调的时候不调整吗

        self.linear = nn.Linear(embedding_size, self.config.vocab_size)
        self.linear.weight = embeddings  # 所以不是共享，只是用于初始化

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, input_tyi, input_attn_mask):
        h = self.bert(input_ids=input_ids, token_type_ids=input_tyi, attention_mask=input_attn_mask)
        out = self.softmax(self.linear(h.last_hidden_state))  # 对数化的概率
        # 直接点积
        if self.is_correct_sent:
            # sent_cls_out = self.sigmoid(self.cls_w(h.pooler_output))
            sent_cls_out = self.cls_w(h.pooler_output)
            seq_pooler_output = self.activation(self.dense(h.last_hidden_state))
            # dense+激活，学习pooler_output
            # seq_cls_out = self.sigmoid(self.cls_w(seq_pooler_output))
            # 后面使用bcewithlogit
            seq_cls_out = self.cls_w(seq_pooler_output)
            return out, sent_cls_out, seq_cls_out
        return out


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

        w = torch.empty(1, embedding_size)
        add_one_weight = nn.init.normal_(w, mean=0.0, std=1.0)

        all_cls_weight = torch.cat([cls_weight, add_one_weight.cuda()], 0)
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

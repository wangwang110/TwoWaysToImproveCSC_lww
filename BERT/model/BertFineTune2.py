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

MASK_TOKEN = 103


def get_mask_ids(input_ids):
    output = torch.ne(input_ids, 0) * MASK_TOKEN
    return output


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

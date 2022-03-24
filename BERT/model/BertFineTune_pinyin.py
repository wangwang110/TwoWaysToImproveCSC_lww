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
    def __init__(self, bert, tokenizer, device, device_ids, num_tones=5, num_inits=23, num_finals=34, num_pinyin=430,use_pinyin=True):
        super(BertFineTune, self).__init__()
        self.device = device
        self.config = bert.config
        embedding_size = self.config.to_dict()['hidden_size']
        self.bert = bert.to(device)
        self.sigmoid = nn.Sigmoid().to(device)

        self.use_pinyin = use_pinyin
        if self.use_pinyin:
            pinyin_size = embedding_size + 16 + 16 + 5
            self.inits_embeddings = nn.Embedding(num_inits, 16)  # 声母
            self.finals_embeddings = nn.Embedding(num_finals, 16)  # 韵母
            self.tones_embeddings = nn.Embedding(num_tones, 5)  # 声调
            self.dense_p = nn.Linear(pinyin_size, pinyin_size)
            self.activation_p = nn.Tanh()
            self.pinyin_linear = nn.Linear(pinyin_size, num_pinyin)

        bert_embedding = bert.embeddings
        word_embeddings_weight = bert.embeddings.word_embeddings.weight
        embeddings = nn.Parameter(word_embeddings_weight, True)
        bert_embedding.word_embeddings = nn.Embedding(self.config.vocab_size, embedding_size, _weight=embeddings)
        self.linear = nn.Linear(embedding_size, self.config.vocab_size)
        self.linear.weight = embeddings

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, input_tyi, input_attn_mask, input_init_ids, input_final_ids, input_tone_ids):
        h = self.bert(input_ids=input_ids, token_type_ids=input_tyi, attention_mask=input_attn_mask)
        out = self.softmax(self.linear(h.last_hidden_state))
        if self.use_pinyin:
            inits_embeddings = self.inits_embeddings(input_init_ids)
            finals_embeddings = self.finals_embeddings(input_final_ids)
            tones_embeddings = self.tones_embeddings(input_tone_ids)
            pinyin_input_embed = torch.cat([h.last_hidden_state, inits_embeddings,
                                            finals_embeddings, tones_embeddings],
                                           2)
            pinyin_hidden = self.activation_p(self.dense_p(pinyin_input_embed))
            pinyin_input_out = self.softmax(self.pinyin_linear(pinyin_hidden))
            # 声母，韵母，音调的向量和字向量拼接预测拼音
            return out, pinyin_input_out
        return out

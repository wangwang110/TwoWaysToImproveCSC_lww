#
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from transformers import BertModel, BertConfig, BertTokenizer
import numpy as np
from tqdm import tqdm

bert_path = "/data_local/plm_models/chinese_L-12_H-768_A-12/"
bert = BertModel.from_pretrained(bert_path, return_dict=True)
tokenizer = BertTokenizer.from_pretrained(bert_path)
config = BertConfig.from_pretrained(bert_path)


def text2vec(src, max_seq_length):
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
        if tok not in tokenizer.vocab:
            tokens[j] = "[UNK]"

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def help_vectorize(batch):
    """

    :param batch:
    :return:
    """
    src_li, trg_li = batch['input'], batch['output']
    max_seq_length = max([len(src) for src in src_li]) + 2
    inputs = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
    outputs = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}

    for src, trg in zip(src_li, trg_li):
        input_ids, input_mask, segment_ids = text2vec(src, max_seq_length)
        inputs['input_ids'].append(input_ids)
        inputs['token_type_ids'].append(segment_ids)
        inputs['attention_mask'].append(input_mask)

        output_ids, output_mask, output_segment_ids = text2vec(trg, max_seq_length)
        outputs['input_ids'].append(output_ids)
        outputs['token_type_ids'].append(output_segment_ids)
        outputs['attention_mask'].append(output_mask)
    return inputs, outputs


all_text_srcs = []
all_text_trgs = []
with open("/data_local/TwoWaysToImproveCSC/BERT/data/wiki_00_csc.train", "r", encoding="utf-8") as f:
    for line in f.readlines():
        src, trg = line.strip().split(" ")
        all_text_srcs.append(src)
        all_text_trgs.append(trg)

for src, trg in tqdm(zip(all_text_srcs, all_text_trgs)):
    tag = 0
    src_tokens = tokenizer.tokenize(src)
    trg_tokens = tokenizer.tokenize(trg)
    for s, t in zip(src_tokens, trg_tokens):
        if s == "[UNK]" and t != "[UNK]":
            tag = 1
            break
    # if tag == 1:
    #     print(src)
    #     print(src_tokens)
    #     print(trg)
    #     print(trg_tokens)

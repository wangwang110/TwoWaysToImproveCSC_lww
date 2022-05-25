import torch
import pickle
import copy
import torch.nn as nn
from loss.focalloss import FocalLoss
from loss.cpoloss import CpoLoss


class BertFineTune(nn.Module):
    def __init__(self, bert, tokenizer, device, device_ids, args):
        super(BertFineTune, self).__init__()

        self.criterion_c = nn.CrossEntropyLoss()

        self.device = device
        self.config = bert.config
        embedding_size = self.config.to_dict()['hidden_size']
        self.bert = bert.to(device)
        self.args = args

        # if self.args.is_multitask:
        hidden_size = self.config.to_dict()['hidden_size']
        self.detection = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid().to(device)

        bert_embedding = bert.embeddings
        word_embeddings_weight = bert.embeddings.word_embeddings.weight  # bert训练好的embeding table
        embeddings = nn.Parameter(word_embeddings_weight, True)  # 参数化
        bert_embedding.word_embeddings = nn.Embedding(self.config.vocab_size,
                                                      embedding_size,
                                                      _weight=embeddings)
        # 原始的bert_embedding.word_embeddings,微调的时候不调整吗?

        if self.args.vocab_refine:
            bert_vocab_refine = pickle.load(open("../large_data/bert_vocab_refine.pkl", "rb"))
            self.cls_size = len(bert_vocab_refine) + 1

            word_ids = [bert_vocab_refine[key] for key in bert_vocab_refine]
            index = torch.tensor(word_ids, device="cuda")

            cls_weight = torch.index_select(word_embeddings_weight, 0, index)  # 用原来的word_embeding初始化

            add_one_weight = word_embeddings_weight[102].view(1, -1)

            all_cls_weight = torch.cat([cls_weight, add_one_weight], 0)
            all_cls_weight = nn.Parameter(all_cls_weight, True)
            # 另外再加一个维度
            self.linear = nn.Linear(embedding_size, self.cls_size)
            self.linear.weight = all_cls_weight  # 共享，只是用于初始化
        else:
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
        copy_text_labels = copy.deepcopy(text_labels)
        if text_labels is not None:
            text_labels[text_labels == 0] = -100
        else:
            text_labels = None

        h = self.bert(input_ids=input_ids, token_type_ids=input_tyi, attention_mask=input_attn_mask)
        out = self.linear(h.last_hidden_state)

        det_out = self.detection(h.last_hidden_state)
        det_prob = self.sigmoid(det_out).squeeze(-1)

        if text_labels is None:
            outputs = (det_prob, out)
        else:
            loss = self.criterion_c(out.transpose(1, 2), text_labels)
            det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid')
            # pad部分不计算损失
            active_mask = input_attn_mask.view(-1, det_out.shape[1]) == 1
            active_probs = det_out.view(-1, det_out.shape[1])[active_mask]
            active_labels = det_labels[active_mask]
            det_loss = det_loss_fct(active_probs, active_labels.float())  # 检测loss（0.7）和纠正loss（0.3）

            if self.args.cpoloss:
                cpo_loss_fct = CpoLoss()
                cpo_loss = cpo_loss_fct(out, copy_text_labels, input_attn_mask)
                outputs = (det_prob, out, cpo_loss, det_loss, loss)
            else:
                outputs = (det_prob, out, det_loss, loss)

        return outputs

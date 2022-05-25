import torch
import pickle
import copy
import torch.nn as nn
from loss.focalloss import FocalLoss
from loss.cpoloss import CpoLoss
from loss.mriginloss import CombineLoss


class BertPredictionWord(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        prediction_scores = self.decoder(hidden_states)
        return prediction_scores


class BertFineTuneMac(nn.Module):
    def __init__(self, bert, tokenizer, device, device_ids, args):
        super(BertFineTuneMac, self).__init__()

        self.criterion_c = nn.CrossEntropyLoss()

        self.device = device
        self.config = bert.config
        self.bert = bert.to(device)
        self.args = args

        # if self.args.is_multitask:
        hidden_size = self.config.to_dict()['hidden_size']
        self.detection = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid().to(device)

        if self.args.vocab_refine:
            bert_vocab_refine = pickle.load(open("../large_data/bert_vocab_refine.pkl", "rb"))
            self.cls_size = len(bert_vocab_refine) + 1

            self.cls = BertPredictionWord(self.config, self.cls_size)

            self.cls.dense.weight = self.bert.cls.predictions.transform.dense.weight
            self.cls.dense.bias = self.bert.cls.predictions.transform.dense.bias
            self.cls.LayerNorm.weight = self.bert.cls.predictions.transform.LayerNorm.weight
            self.cls.LayerNorm.bias = self.bert.cls.predictions.transform.LayerNorm.bias

            word_ids = [bert_vocab_refine[key] for key in bert_vocab_refine]
            index = torch.tensor(word_ids, device="cuda")

            decoder_weight = torch.index_select(self.bert.cls.predictions.decoder.weight, 0, index)
            decoder_bias = torch.index_select(self.bert.cls.predictions.decoder.bias, 0, index).view(-1, 1)

            add_one_weight = self.bert.cls.predictions.decoder.weight[102].view(1, -1)
            add_one_bias = self.bert.cls.predictions.decoder.bias[102].view(1, -1)

            all_decoder_weight = torch.cat([decoder_weight, add_one_weight], 0)
            all_decoder_weight = nn.Parameter(all_decoder_weight, True)

            all_decoder_bias = torch.cat([decoder_bias, add_one_bias], 0)
            all_decoder_bias = nn.Parameter(all_decoder_bias.view(-1), True)
            # 另外再加一个维度

            self.cls.decoder.weight = all_decoder_weight
            self.cls.decoder.bias = all_decoder_bias
        else:
            self.cls = BertPredictionWord(self.config, self.config.vocab_size)
            self.cls.dense.weight = self.bert.cls.predictions.transform.dense.weight
            self.cls.dense.bias = self.bert.cls.predictions.transform.dense.bias
            self.cls.LayerNorm.weight = self.bert.cls.predictions.transform.LayerNorm.weight
            self.cls.LayerNorm.bias = self.bert.cls.predictions.transform.LayerNorm.bias
            self.cls.decoder.weight = self.bert.cls.predictions.decoder.weight
            self.cls.decoder.bias = self.bert.cls.predictions.decoder.bias

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

        bert_outputs = self.bert(input_ids=input_ids, token_type_ids=input_tyi, attention_mask=input_attn_mask,
                                 labels=text_labels, return_dict=True, output_hidden_states=True)

        det_out = self.detection(bert_outputs.hidden_states[-1])
        det_prob = self.sigmoid(det_out).squeeze(-1)

        # out = bert_outputs.logits
        out = self.cls(bert_outputs.hidden_states[-1])

        if text_labels is None:
            outputs = (det_prob, out)
        else:
            # loss = bert_outputs.loss
            if self.args.error_weight != 1.0:
                loss_fct = CombineLoss(self.args.error_weight)
                loss = loss_fct(out, input_ids, copy_text_labels)
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

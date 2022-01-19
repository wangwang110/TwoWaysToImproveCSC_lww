# Copyright 2018 CVTE . All Rights Reserved.
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from transformers import BertModel, BertConfig, BertTokenizer
import numpy as np

batch = {'input': ['最后希望您能够听到居民们的心听，尽快给我们一个合理的处里方案，还给居民一个安静的生活环境。', '反过来说，他也给了我一些很重要的教育：怎么党人，什么种生活才好。', '面对薪水减少、也很可能会失业的危险，物价却没有下跌，国民的生活一天比一天要紧裤]。', '只要更多的人支持职业妇女让他继续工作，才能实现理想的社会。', '虽然我以前在咖啡店里，听到她的歌，可是不知道它的名字，所以没办法找买。', '我最近忙著准备学校的活动。因为有那场活动，所以我觉得很累。不过我国得还蛮充实的生活。', '因为少年人对国家、社会或家庭来说非常的宝贵、非常的重要、非常的需要。', '他没有做坏的事情，可是他被警察瓜了，而且很久不能见面跟他的家人。', '秋天的到来，令人联想到的就是树叶变色，并且掉落满地。使得扫落叶的人怎么扫都扫不完，有的人或许也会心想：秋天就竟要到何时才会结束呢？', '首先，很抱羡已经跟你说这种实话实说的我，最近李先生的厂长发生两件事情，不仅令我的家人觉得很不方便，连我的邻居也觉得非常懊恼。', '放假快要来了，暱友打算吗？', '如果你跟朋友要去晚，你要早一点回家。', '我会选著房子在板桥市是因为房价在板桥市比在台北市查多了。我现在的套房六乘六大，有自己的床、桌椅、电视，包括水电，每个月只要６０００元。', '我们杂志的首要语言是英文，从而稳定的英文非常重要。接著，新人也得证明他的中文能力相当于国际和与水平能力测验的最高级程度。', '走路的时候他试试看厅路上的汽车，就一位先生厅还告诉对我弟弟，他也到英国去，所以我弟弟可以跟他一起走。', '乌来的花节才刚开始，不过，从乌来的网站来看，现在已经有五花八们的花开了。', '作者刘墉小时后，家里陷入火海，虽然逃出来了，但眉毛、睫毛都烧光了，他默默接受，并不认为这是「挫折」，只能算「遭遇」。', '世界上唯一不会离你而去的就是你的亲人，不管你变的怎样，面临什么危险他们都会在那陪著你直到最后。', '那个课有一点容易。而且我的中文水平比同学们的高。我先跟老师说了，再决定换班了。然后我坐第三课，所以我一定买第三本书。', '到现在我还是记得一清二楚爸爸的话。就是人一定要努力，不官你做什么事情不可失去了斗志。'], 'output': ['最后希望您能够听到居民们的心听，尽快给我们一个合理的处理方案，还给居民一个安静的生活环境。', '反过来说，他也给了我一些很重要的教育：怎么当人，什么种生活才好。', '面对薪水减少、也很可能会失业的危险，物价却没有下跌，国民的生活一天比一天要辛苦]。', '只要更多的人支持职业妇女让她继续工作，才能实现理想的社会。', '虽然我以前在咖啡店里，听到她的歌，可是不知道她的名字，所以没办法找买。', '我最近忙著准备学校的活动。因为有那场活动，所以我觉得很累。不过我过得还蛮充实的生活。', '因为少年人对国家、社会或家庭来说非常地宝贵、非常地重要、非常地需要。', '他没有做坏的事情，可是他被警察抓了，而且很久不能见面跟他的家人。', '秋天的到来，令人联想到的就是树叶变色，并且掉落满地。使得扫落叶的人怎么扫都扫不完，有的人或许也会心想：秋天究竟要到何时才会结束呢？', '首先，很抱歉已经跟你说这种实话实说的我，最近李先生的厂长发生两件事情，不仅令我的家人觉得很不方便，连我的邻居也觉得非常懊恼。', '放假快要来了，你有打算吗？', '如果你跟朋友要去玩，你要早一点回家。', '我会选择房子在板桥市是因为房价在板桥市比在台北市差多了。我现在的套房六乘六大，有自己的床、桌椅、电视，包括水电，每个月只要６０００元。', '我们杂志的首要语言是英文，从而稳定的英文非常重要。接著，新人也得证明他的中文能力相当于国际汉语水平能力测验的最高级程度。', '走路的时候他试试看听路上的汽车，就一位先生停还告诉对我弟弟，他也到英国去，所以我弟弟可以跟他一起走。', '乌来的花节才刚开始，不过，从乌来的网站来看，现在已经有五花八门的花开了。', '作者刘墉小时候，家里陷入火海，虽然逃出来了，但眉毛、睫毛都烧光了，他默默接受，并不认为这是「挫折」，只能算「遭遇」。', '世界上唯一不会离你而去的就是你的亲人，不管你变得怎样，面临什么危险他们都会在那陪著你直到最后。', '那个课有一点容易。而且我的中文水平比同学们的高。我先跟老师说了，再决定换班了。然后我做第三课，所以我一定买第三本书。', '到现在我还是记得一清二楚爸爸的话。就是人一定要努力，不管你做什么事情不可失去了斗志。']}

batch = {'input': ['我们会有单心和问题。要不然我们会告诉市长，让他帮忙。', '他穿蓝色的牛仔裤根背一个黑色的背包。', '我们约在台北１０１前面，好不好？六点可以马？请你星期六以前给我打电话。', '同时青年的减少会造成国家长其人口下降，也就是说国家的税收会减少。国家没有足够得钱去做公共建设，使得商业活动会越来越少，国际竞争力下降。', '虽然我多理解你的想法，但我个人已经有了自己的人生计划。', '希望后天我会上可了。', '有一些人就一直不去为自己未来的方向努力，等到了成人，因为之前不努力，所以就要看命运的安排和自已的努力，这就是由命运来决定未来。', '王先生跟他说好多话，他们一直说话一直笑很开兴了！', '我对教室里该不该装录影机这个议题表我的看话是．．．我同意校长想的办法。', '最后希望您能够听到居民们的心听，尽快给我们一个合理的处里方案，还给居民一个安静的生活环境。', '听了一首歌，我就下决定要买她的ＣＤ，她还说要是又谁买或有她的ＣＤ。当天他自亲签名，而且前一百名还会赠送海报。', '上课开始时，今天帮我忙的女生就到教室近来了。', '因为小安没人琣他，他有一点不高兴。', '我最近想了很久，总于想出来把问题解决的方法。', '有１天我跟同学们一起去逛街跟很多同学，那天是礼拜６我们从早上我们礿好了１０点在学校门口间大家都很准时，我们坐工车到新竹市，车停的时候大家都以为是到了！可是不是每个人脸上都很开心。', '他们花了一天再看那个球赛。', '最近我听说我孩子的小学的校长只是在每个教室里装了影几，我觉得有点担心。', '做父母的人比较容易管理他们的孩子，有些孩子往往娇课。', '我作天学中文到晚上十点。', '另外一件事就是为了毕业写的那边文论，一般来说写文论很难，可是因为她推荐很好所以我写了一篇非常棒的文论。'], 'output': ['我们会有担心和问题。要不然我们会告诉市长，让他帮忙。', '他穿蓝色的牛仔裤跟背一个黑色的背包。', '我们约在台北１０１前面，好不好？六点可以吗？请你星期六以前给我打电话。', '同时青年的减少会造成国家长期人口下降，也就是说国家的税收会减少。国家没有足够的钱去做公共建设，使得商业活动会越来越少，国际竞争力下降。', '虽然我都理解你的想法，但我个人已经有了自己的人生计划。', '希望后天我会上课了。', '有一些人就一直不去为自己未来的方向努力，等到了成人，因为之前不努力，所以就要看命运的安排和自己的努力，这就是由命运来决定未来。', '王先生跟他说好多话，他们一直说话一直笑很开心了！', '我对教室里该不该装录影机这个议题表我的看法是．．．我同意校长想的办法。', '最后希望您能够听到居民们的心听，尽快给我们一个合理的处理方案，还给居民一个安静的生活环境。', '听了一首歌，我就下决定要买她的ＣＤ，她还说要是有谁买或有她的ＣＤ。当天他自亲签名，而且前一百名还会赠送海报。', '上课开始时，今天帮我忙的女生就到教室进来了。', '因为小安没人陪他，他有一点不高兴。', '我最近想了很久，终于想出来把问题解决的方法。', '有１天我跟同学们一起去逛街跟很多同学，那天是礼拜６我们从早上我们约好了１０点在学校门口见大家都很准时，我们坐公车到新竹市，车停的时候大家都以为是到了！可是不是每个人脸上都很开心。', '他们花了一天在看那个球赛。', '最近我听说我孩子的小学的校长只是在每个教室里装了影机，我觉得有点担心。', '做父母的人比较容易管理他们的孩子，有些孩子往往翘课。', '我昨天学中文到晚上十点。', '另外一件事就是为了毕业写的那篇文论，一般来说写文论很难，可是因为她推荐很好所以我写了一篇非常棒的文论。']}
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


inputs = tokenizer(batch['input'], padding=True, truncation=True, return_tensors="pt")
inputs_s, _ = help_vectorize(batch)

for i in range(20):
    if sum(inputs["input_ids"][i]) != sum(inputs_s["input_ids"][i]):
        print(batch['input'][i])
        print(inputs["input_ids"][i])
        for s in inputs["input_ids"][i]:
            print(s)
        print([tokenizer.convert_ids_to_tokens(s.item()) for s in inputs["input_ids"][i]])
        # print(inputs_s["input_ids"][i])
        print([tokenizer.convert_ids_to_tokens(s) for s in inputs_s["input_ids"][i]])
        print("==================")

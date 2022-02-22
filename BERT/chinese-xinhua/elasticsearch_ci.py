# -*- coding: utf-8 -*-

import time
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import json
import re
import spacy
import os
import pickle
from pypinyin import pinyin, lazy_pinyin, Style

nlp = spacy.load("zh_core_web_sm")

# 候选集
filepath = '/data_local/TwoWaysToImproveCSC/BERT/save/confusion.file'
with open(filepath, 'rb') as f:
    confusion_set = pickle.load(f)


class ElasticSearch:
    def __init__(self):
        self.es = Elasticsearch([{"host": "10.21.2.35", "port": 9201}])
        # ip (不要加http)
        # port

    def batch_data(self, ):
        """ 批量写入数据 """
        start = time.time()
        vocab_dict = pickle.load(open("./vocab_xiaoxue.pkl", "rb"))
        chengyu_li = []
        for s in vocab_dict:
            if 2 <= len(s) < 4 and not re.search('[^\u4e00-\u9fa5]', s):
                # pos = [str(p) for p in range(len(s))]
                # ci_pos_li = [w + "_" + p for w, p in zip(list(s), pos)]
                # ci_pos_str = " ".join(ci_pos_li)
                pinyin = " ".join(lazy_pinyin(s))
                chengyu_li.append((s, vocab_dict[s], pinyin))

        print(len(chengyu_li))  # 264434

        # 写入数据
        action = ({
            "_index": "ci_v4",
            "_type": "_doc",
            "_id": i,
            "_source": {
                'ci': chengyu_li[i][0],  # 词语,
                'fre': chengyu_li[i][1],  # 词频,
                'pinyin': chengyu_li[i][2],
                #  'ci_pos': chengyu_li[i][3]
            }
        } for i in range(len(chengyu_li)))

        helpers.bulk(self.es, action)
        print('批量导入索引 共耗时约 {:.2f} 秒'.format(time.time() - start))
        return 0

    def search(self, query_tran='白费'):
        """
        输入四个字的词
        :param query_tran:
        :return:
        """
        num = len(query_tran)

        query = {
            "query": {
                "bool": {
                    "must": [{
                        "match": {
                            "ci": {
                                "query": query_tran,
                                "minimum_should_match": num - 1
                            }
                        }}
                    ],
                    "should": [
                        {
                            "match_phrase": {
                                "pinyin": {
                                    "query": " ".join(lazy_pinyin(query_tran))
                                }
                            }}
                    ]
                }
            },
            "highlight": {
                "fields": {
                    "ci": {}
                }
            }
        }

        result = self.es.search(index="ci_v4", body=query, size=10)
        hits_list = result['hits']['hits']
        str_li = []
        for hit in hits_list:
            count = 0
            if len(hit['_source']['ci']) != len(query_tran):
                continue
            for s, t in zip(query_tran, hit['_source']['ci']):
                if s == t:
                    count += 1

            if query_tran == hit['_source']['ci']:
                str_li = [(query_tran, query_tran)]
                return str_li
            elif count == num - 1:
                pattern = ""
                match_character_li = re.findall("<em>(.*?)</em>", hit["highlight"]["ci"][0])
                for w in hit['_source']['ci']:
                    if w not in match_character_li:
                        pattern += "."
                    else:
                        pattern += w
                str_li.append((pattern, hit['_source']['ci']))
                # 匹配模式，匹配到的成语
        return str_li

    def delete_all(self, ):
        start = time.time()
        delete_by_all = {"query": {"match_all": {}}}
        # self.es.delete_by_query(index="ci_v1", body=delete_by_all)
        # self.es.delete_by_query(index="ci_v2", body=delete_by_all)
        self.es.delete_by_query(index="ci_v4", body=delete_by_all)
        print('删除全部索引 共耗时约 {:.2f} 秒'.format(time.time() - start))


def find_most_similar(src, trgs):
    """
    如果不用语言模型，为了更好的保持高准确度
    既要拼音一致，又要在候选集？
    :param src:
    :param trgs:
    :return:
    """
    for trg in trgs:
        for s, t in zip(list(src), list(trg)):
            if s != t:
                if lazy_pinyin(s) == lazy_pinyin(t) and s in confusion_set and t in confusion_set[s]:
                    print("++拼音一致++", trg)
                    return trg
                elif (set(lazy_pinyin(s)[0]) - set(lazy_pinyin(t)[0]) == set(["g"]) or
                      set(lazy_pinyin(t)[0]) - set(lazy_pinyin(s)[0]) == set(["g"])) \
                        and s in confusion_set and t in confusion_set[s]:
                    print("++前后鼻不分++", trg)
                    return trg
                else:
                    if s not in confusion_set:
                        return src
                    if t in confusion_set[s]:
                        print("++在混淆集++", trg)
                        return trg
    return src


def correct_sent(text):
    """
    纠正输入文本 text
    :param text:
    :return:
    """
    words = []
    words_tag = []
    for w in nlp(text):
        words.append(w.text)
        words_tag.append(w.tag_)

    # 字数大于1的词语
    four_words = []
    num = len(words)
    for i in range(num - 1):
        if len(words[i]) > 1:
            four_words.append(words[i])

    correct = {}
    # print(four_words)
    for src in four_words:
        res_li = elastic_search.search(query_tran=src)
        trgs = []
        for res in res_li:
            pattern, trg = res
            if src == trg:
                # print("发现成语：", trg)
                break
            else:
                match_idom = re.search(pattern, text)
                if match_idom is not None:
                    trgs.append(trg)
        if len(trgs) >= 1:
            cor = find_most_similar(src, trgs)
            correct[src] = cor
    print(correct)
    for item in correct:
        text = text.replace(item, correct[item])
    return text


if __name__ == '__main__':
    elastic_search = ElasticSearch()
    elastic_search.delete_all()
    elastic_search.batch_data()

    # # 测试词语纠错
    text_li = [
        "大自然也一样的，无法天天都是晴天，天天都很顺利，但生活，就是如此这般，有失才有得，只是每个人是如何去看侍的。",
        "在前往拥抱梦想的过程中，往往都是一条充满毒蛇猛兽，陷井重重的道路，当你一但经历和刻服了这些关卡时，就是你张开双手迎接成功的时候。",
        "有时候人会因为习管了身边的一切事物，而渐渐的认为一切都是理所当然的，开始不珍惜或是糟踏自己所理想当然的事物，直到有一天它逍失了，才感觉到它的重要性，才了解到没有了它是多么不方便。",
        "另外，还挂著个小时钟，能够提醒大家在当下所该做的事，时光是不可回朔的，因此它珍贵过了一切，小小的时钟告诉了我时刻，也时时叮咛我把握当下。",
        "我认为，「方向」和「梦想」有些相似。只要朝著自己的方向前进，同样的，你也正朝著自己的梦想走。所以，人一定要有志向，才不会白废自己的生命。",
        "已经开了两个多小时的会议，双方代表还是莫衷一事，没有定论。"
    ]
    for sent in text_li:
        new_sent = correct_sent(sent)
        print(sent)
        print(new_sent)
        print("\n")

    # path = "/data_local/TwoWaysToImproveCSC/BERT/data/13test.txt"
    # with open(path, "r", encoding="utf-8") as f, open("../data/13test_ci.txt", "w", encoding="utf-8") as fw:
    #     id = 1
    #     for line in f.readlines():
    #         print("===id===:{}\n".format(str(id)))
    #         sent, sent_trg = line.strip().split(" ")
    #         new_sent = correct_sent(sent.strip())
    #
    #         if sent != new_sent:
    #             print(sent)
    #             print(new_sent)
    #             print(sent_trg)
    #             print("===============\n")
    #             fw.write(new_sent + " " + sent_trg + "\n")
    #         else:
    #             fw.write(sent + " " + sent_trg + "\n")
    #
    #         id += 1
    #
    # path = "/data_local/TwoWaysToImproveCSC/BERT/cc_data/chinese_spell_4.txt"
    # with open(path, "r", encoding="utf-8") as f, open("../cc_data/chinese_spell_ci_4.txt", "w",
    #                                                   encoding="utf-8") as fw:
    #     id = 1
    #     for line in f.readlines():
    #         print("===id===:{}\n".format(str(id)))
    #         sent, sent_trg = line.strip().split(" ")
    #         new_sent = correct_sent(sent.strip())
    #
    #         if sent != new_sent:
    #             print(sent)
    #             print(new_sent)
    #             print(sent_trg)
    #             print("===============\n")
    #             fw.write(new_sent + " " + sent_trg + "\n")
    #         else:
    #             fw.write(sent + " " + sent_trg + "\n")
    #
    #         id += 1

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

# path = "/data_local/TwoWaysToImproveCSC/large_data/char_pic/"
# bin_path = os.path.dirname(path) + "/character.bin"
# img_vector_dict = pickle.load(open(bin_path, "rb"))

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
        fliter_set = set(["再接再历", "号啕大哭", "按纳不住", "水楔不通"])
        save_set = set()

        start = time.time()
        path = "/data_local/TwoWaysToImproveCSC/BERT/chinese-xinhua/data/idiom.json"
        chengyu_li = []
        with open(path, 'r', encoding="utf-8") as f:
            chengyu_dict = json.load(f)
            for item in chengyu_dict:
                if item['word'] in fliter_set or item['word'] in save_set:
                    continue
                save_set.add(item['word'])
                chengyu_li.append(item)

        path_oulu = "/data_local/TwoWaysToImproveCSC/BERT/chinese-xinhua/idom_detail.pkl"
        chengyu_oulu = pickle.load(open(path_oulu, "rb"))
        for item in chengyu_oulu:
            if item["成语"] in fliter_set or item["成语"] in save_set:
                continue
            else:
                save_set.add(item['成语'])
                new_item = {}
                new_item["word"] = item['成语']
                new_item["derivation"] = item['出处']
                new_item["explanation"] = item['解释']
                new_item["example"] = item['示例']
                chengyu_li.append(new_item)

        print(len(chengyu_li))  # 49550

        # 写入数据
        action = ({
            "_index": "idom_v1",
            "_type": "_doc",
            "_id": i,
            "_source": {
                'word': chengyu_li[i]['word'],  # 成语,
                'derivation': chengyu_li[i]['derivation'],  # 出处
                'explanation': chengyu_li[i]['explanation'],  # 解释
                'example': chengyu_li[i]['example'],  # 示例
            }
        } for i in range(len(chengyu_li)))

        helpers.bulk(self.es, action)
        print('批量导入索引 共耗时约 {:.2f} 秒'.format(time.time() - start))
        return 0

    def search(self, query_tran='柳暗花明'):
        """
        输入四个字的词
        :param query_tran:
        :return:
        """
        query = {
            "query": {
                "match": {
                    "word": {
                        "query": query_tran,
                        "minimum_should_match": "75%"
                    }
                }

            },
            "highlight": {
                "fields": {
                    "word": {}
                }
            }
        }

        result = self.es.search(index="idom_v1", body=query, size=5)
        hits_list = result['hits']['hits']
        str_li = []
        for hit in hits_list:
            count = 0
            for s, t in zip(query_tran, hit['_source']['word']):
                if s == t:
                    count += 1

            if count == 4:
                str_li = [(query_tran, query_tran)]
                return str_li
            elif count == 3:
                pattern = ""
                match_character_li = re.findall("<em>(.*?)</em>", hit["highlight"]["word"][0])
                for w in hit['_source']['word']:
                    if w not in match_character_li:
                        pattern += "."
                    else:
                        pattern += w
                str_li.append((pattern, hit['_source']['word']))
                # 匹配模式，匹配到的成语
        return str_li

    # def delete_all(self, ):
    #     start = time.time()
    #     # delete_by_all = {"query": {"match_all": {}}}
    #     # self.es.delete_by_query(index="poem", body=delete_by_all)
    #     # self.es.delete_by_query(index="test", body=delete_by_all)
    #     # self.es.delete_by_query(index="poem_test", body=delete_by_all)
    #     # self.es.delete_by_query(index="poem_v4", body=delete_by_all)
    #     print('删除全部索引 共耗时约 {:.2f} 秒'.format(time.time() - start))


# # 计算两个向量之间的余弦相似度
# def cosine_similarity(vector1, vector2):
#     dot_product = 0.0
#     normA = 0.0
#     normB = 0.0
#     for a, b in zip(vector1, vector2):
#         dot_product += a * b
#         normA += a ** 2
#         normB += b ** 2
#     if normA == 0.0 or normB == 0.0:
#         return 0
#     else:
#         return dot_product / ((normA ** 0.5) * (normB ** 0.5))
#

# def find_most_similar(src, trgs):
#     """
#     如果有拼音相同的直接返回
#     :param src:
#     :param trgs:
#     :return:
#     """
#     scores = []
#     for trg in trgs:
#         for s, t in zip(list(src), list(trg)):
#             if s != t:
#                 if lazy_pinyin(s) == lazy_pinyin(t):
#                     print("++拼音一致++", trg)
#                     return trg
#                 else:
#                     if s not in img_vector_dict:
#                         return src
#                     if t not in img_vector_dict:
#                         continue
#                     scores.append((trg, cosine_similarity(img_vector_dict[s], img_vector_dict[t])))
#     if len(scores) >= 1:
#         sorted_score = sorted(scores, key=lambda s: s[1], reverse=True)
#         print(sorted_score)
#         return sorted_score[0][0]
#
#     return src


def find_most_similar(src, trgs):
    """
    如果有拼音相同的 或者 在候选集中的
    :param src:
    :param trgs:
    :return:
    """
    print(trgs)
    for trg in trgs:
        for s, t in zip(list(src), list(trg)):
            if s != t:
                if lazy_pinyin(s) == lazy_pinyin(t):
                    print("++拼音一致++", trg)
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
    # print(words)

    # 前后组成4字候选成语，要求词性一致
    four_words = []
    num = len(words)
    for i in range(num - 1):
        if len(words[i]) == 4:
            four_words.append(words[i])
        else:
            j = i + 1
            # words_tag[i] == words_tag[j] and
            if len(words[i] + words[j]) == 4:
                four_words.append(words[i] + words[j])

    correct = {}
    # print(four_words)
    for src in four_words:
        res_li = elastic_search.search(query_tran=src)
        trgs = []
        for res in res_li:
            pattern, trg = res
            if pattern == trg:
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
    elastic_search.batch_data()
    # elastic_search.delete_all()

    # # 测试成语纠错
    text_li = [
               "有时候人会因为习惯了身边的一切事物，而渐渐的认为一切都是理所当然的，开始不珍惜或是糟踏自己所理想当然的事物，直到有一天它逍失了，才感觉到它的重要性，才了解到没有了它是多么不方便。",
               "想不到他监守自盗，偷天换日，真是胆大忘为。",
               "互助组应当帮助鳏寡孤读",
               "已经开了两个多小时的会议，双方代表还是莫衷一事，没有定论。"
               ]
    for sent in text_li:
        new_sent = correct_sent(sent)
        print(sent)
        print(new_sent)
        print("\n")

    path = "/data_local/TwoWaysToImproveCSC/BERT/data/13test.txt"
    with open(path, "r", encoding="utf-8") as f, open("../data/13test_cy_new.txt", "w", encoding="utf-8") as fw:
        id = 1
        for line in f.readlines():
            print("===id===:{}\n".format(str(id)))
            sent, sent_trg = line.strip().split(" ")
            new_sent = correct_sent(sent.strip())

            if sent != new_sent:
                print(sent)
                print(new_sent)
                print(sent_trg)
                print("===============\n")
                fw.write(new_sent + " " + sent_trg + "\n")
            else:
                fw.write(sent + " " + sent_trg + "\n")

            id += 1

    path = "/data_local/TwoWaysToImproveCSC/BERT/cc_data/chinese_spell_4.txt"
    with open(path, "r", encoding="utf-8") as f, open("../cc_data/chinese_spell_cy_new_4.txt", "w",
                                                      encoding="utf-8") as fw:
        id = 1
        for line in f.readlines():
            print("===id===:{}\n".format(str(id)))
            sent, sent_trg = line.strip().split(" ")
            new_sent = correct_sent(sent.strip())

            if sent != new_sent:
                print(sent)
                print(new_sent)
                print(sent_trg)
                print("===============\n")
                fw.write(new_sent + " " + sent_trg + "\n")
            else:
                fw.write(sent + " " + sent_trg + "\n")

            id += 1

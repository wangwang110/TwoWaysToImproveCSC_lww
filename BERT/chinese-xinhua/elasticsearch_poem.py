# -*- coding: utf-8 -*-

import time
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import json
import pandas as pd
import re
import pickle
from pypinyin import pinyin, lazy_pinyin, Style

# 候选集
filepath = '/data_local/TwoWaysToImproveCSC/BERT/save/confusion.file'
with open(filepath, 'rb') as f:
    confusion_set = pickle.load(f)


# 作者 + 题目，用于去重
class ElasticSearch:
    def __init__(self):
        self.es = Elasticsearch([{"host": "10.21.2.35", "port": 9201}])

    def batch_data(self, ):
        """ 批量写入数据 """
        start = time.time()
        data = pd.read_hdf("./poem_retrieve.hdf")
        data.columns = ['title', 'author', 'dynasty', 'content_seg', 'sgl_cont']
        data = data.drop_duplicates(subset=['sgl_cont', 'title'], keep='first', inplace=False)
        data.index = range(len(data))

        action = ({
            "_index": "poem_v1",
            "_type": "_doc",
            "_id": i,
            "_source": {
                'title': data['title'][i],  # 诗的题目
                'sgl_cont': data['sgl_cont'][i],  # 诗中的某一句
                'author': data['author'][i],  # 诗的作者
                'dynasty': data['dynasty'][i],  # 诗的朝代
                'content_seg': data['content_seg'][i],  # 诗的整体内容

            }
        } for i in range(len(data)))

        helpers.bulk(self.es, action)
        print('批量导入索引 共耗时约 {:.2f} 秒'.format(time.time() - start))
        return 0

    def search(self, query_tran='天生我材必有用'):
        # https://www.elastic.co/guide/cn/elasticsearch/guide/current/match-multi-word.html
        # https://www.elastic.co/guide/cn/elasticsearch/guide/current/proximity-relevance.html
        query = {
            "query": {
                "match": {
                    "sgl_cont": {
                        "query": query_tran,
                        "minimum_should_match": 5
                        # "fuzziness": "AUTO"
                    }
                }

            },
            "highlight": {
                "fields": {
                    "sgl_cont": {}
                }
            }
        }

        result = self.es.search(index="poem_v1", body=query, size=1)
        hits_list = result['hits']['hits']
        str_li = []
        for hit in hits_list:
            # print(hit)
            pattern = ""
            match_character_li = re.findall("<em>(.*?)</em>", hit["highlight"]["sgl_cont"][0])
            if len(hit['_source']['sgl_cont']) - len(match_character_li) <= 4:
                for w in hit['_source']['sgl_cont']:
                    if w not in match_character_li:
                        pattern += "."
                    else:
                        pattern += w
                str_li.append((pattern, hit['_source']['sgl_cont']))
                # 匹配模式，匹配到的成语
        return str_li

    def delete_all(self, ):
        start = time.time()
        # delete_by_all = {"query": {"match_all": {}}}
        # self.es.delete_by_query(index="poem", body=delete_by_all)
        # self.es.delete_by_query(index="test", body=delete_by_all)
        # self.es.delete_by_query(index="poem_test", body=delete_by_all)
        # self.es.delete_by_query(index="poem_v4", body=delete_by_all)
        print('删除全部索引 共耗时约 {:.2f} 秒'.format(time.time() - start))


def remove_some_correct(src, trg):
    """
    如果有拼音相同的 或者 在候选集中的
    :param src:
    :param trgs:
    :return:
    """
    res = ""
    for s, t in zip(list(src), list(trg)):
        if s != t and lazy_pinyin(s) == lazy_pinyin(t):
            print("++拼音一致++", trg)
            res += t
        elif s != t and s in confusion_set and t in confusion_set[s]:
            print("++在混淆集++", trg)
            res += t
        else:
            res += s

    return res


def correct_sent(text):
    res_li = elastic_search.search(query_tran=text)
    # print(res_li)
    if len(res_li) == 0:
        return text

    # 已经匹配到的位置，不再处理。
    # 防止将本身正确的改错，偷天换日--移天换日
    correct = {}
    for res in res_li:
        pattern, trg = res
        if pattern == trg:
            pass
            # print("发现诗句：", trg)
        else:
            match_idom = re.search(pattern, text)
            if match_idom is not None:
                s, t = match_idom.span()
                src = text[s:t]
                res = remove_some_correct(src, trg)
                correct[src] = res
                # print("发现诗句=：", trg)
    for item in correct:
        text = text.replace(item, correct[item])
    return text


if __name__ == '__main__':
    elastic_search = ElasticSearch()
    # elastic_search.batch_data()
    # elastic_search.delete_all()

    # 测试成语纠错
    # text = "古人曾说：「好鸟枝头奕朋友，落花水面皆文章。」、「万物静观皆自得。」在繁忙的生活中，不妨放慢脚步，好好「静观」万物，享受快乐吧！"
    text = "「天将降大任于斯人也，必先苦其心智，劳其筋骨」在暴风中成长的初芽更为茁壮，在豪雨中扎根的枝干更为坚固。 "
    new_text = correct_sent(text)
    print(text)
    print(new_text)

    path = "/data_local/TwoWaysToImproveCSC/BERT/data/13test.txt"
    with open(path, "r", encoding="utf-8") as f, open("./sighan13.out", "w", encoding="utf-8") as fw:
        for line in f.readlines():
            print(line)
            line, trg = line.strip().split(" ")
            new_line = correct_sent(line)
            # fw.write(new_line + "\n")
            if line != new_line:
                print(line)
                print(new_line)
                print(trg)
                print("=============================\n")

    path = "/data_local/TwoWaysToImproveCSC/BERT/cc_data/chinese_spell_4.txt"
    with open(path, "r", encoding="utf-8") as f, open("./sighan13.out", "w", encoding="utf-8") as fw:
        for line in f.readlines():
            print(line)
            line, trg = line.strip().split(" ")
            new_line = correct_sent(line)
            # fw.write(new_line + "\n")
            if line != new_line:
                print(line)
                print(new_line)
                print(trg)
                print("=============================\n")

# -*- coding: utf-8 -*-

import time
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import json
import pandas as pd
import re
import pickle


# 名人名言中的诗词要去掉

class ElasticSearch:
    def __init__(self):
        self.es = Elasticsearch([{"host": "10.21.2.35", "port": 9201}])

    def batch_data(self, ):
        """ 批量写入数据 """
        start = time.time()
        # 读入名人名言
        paths = [
            "./xiaoxue_mrmy.txt",
            "./baike_mrmy.txt",
        ]
        mrmy_dict = {}
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    items = line.strip().split("——")
                    if len(items) > 2:
                        continue
                    elif len(items) == 2:
                        if len(items[1]) < 8:
                            # 姓名
                            mrmy_dict[items[0]] = items[1]
                    else:
                        if items[0] not in mrmy_dict:
                            mrmy_dict[items[0]] = ""
        # 读入诗句
        data = pd.read_hdf("./poem_retrieve.hdf")
        data.columns = ['title', 'author', 'dynasty', 'content_seg', 'sgl_cont']
        data = data.drop_duplicates(subset=['sgl_cont', 'title'], keep='first', inplace=False)
        data.index = range(len(data))

        fliter_set = set()
        for i in range(len(data)):
            fliter_set.add(data['sgl_cont'][i])

        # 将名人名言，在诗句中的去掉
        # 总共4263句
        mrmy_li = []
        for sent in mrmy_dict:
            if sent in fliter_set:
                continue
            mrmy_li.append((sent, mrmy_dict[sent]))

        print(len(mrmy_li))

        action = ({
            "_index": "mrmy_v1",
            "_type": "_doc",
            "_id": i,
            "_source": {
                'content': mrmy_li[i][0],  # 名人名言内容
                'author': mrmy_li[i][1]  # 名人名言作者
            }
        } for i in range(len(mrmy_li)))

        helpers.bulk(self.es, action)
        print('批量导入索引 共耗时约 {:.2f} 秒'.format(time.time() - start))
        return 0

    def search(self, query_tran='时间就是生命'):
        query = {
            "query": {
                "match": {
                    "content": {
                        "query": query_tran,
                        "minimum_should_match": 6
                    }
                }
            },
            "highlight": {
                "fields": {
                    "content": {}
                }
            }
        }

        result = self.es.search(index="mrmy_v1", body=query, size=5)
        # 默认一个输入最多只包含1个诗句
        hits_list = result['hits']['hits']
        str_li = []
        for hit in hits_list:
            match_str = hit['_source']['content']
            match_character_li = re.findall("<em>(.*?)</em>", hit["highlight"]["content"][0])
            if len(match_str) - len(match_character_li) <= 4:
                # 与原来的数据差别小于4个字，包括标点
                str_li.append(hit['_source'])
        return str_li

    def delete_all(self, ):
        start = time.time()
        delete_by_all = {"query": {"match_all": {}}}
        self.es.delete_by_query(index="mrmy_v1", body=delete_by_all)
        print('删除全部索引 共耗时约 {:.2f} 秒'.format(time.time() - start))


if __name__ == '__main__':
    elastic_search = ElasticSearch()
    # elastic_search.delete_all()
    # elastic_search.batch_data()

    res = elastic_search.search("时间就是生命")
    print(res)

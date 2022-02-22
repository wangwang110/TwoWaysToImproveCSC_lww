# -*- coding: utf-8 -*-

import time
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import json
import pandas as pd
import kenlm
import spacy
import re
import os
import pickle
from pypinyin import pinyin, lazy_pinyin, Style

"""
定义一些拼音错误的规则
1.前后鼻音不分
2.平翘舌部分
3. 字符图片相似度
"""


# 计算两个向量之间的余弦相似度
def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return dot_product / ((normA ** 0.5) * (normB ** 0.5))


class CscMatch:
    def __init__(self):
        self.es = Elasticsearch([{"host": "10.21.2.35", "port": 9201}])
        # 候选集
        filepath = '/data_local/TwoWaysToImproveCSC/BERT/save/confusion.file'
        with open(filepath, 'rb') as f:
            self.confusion_set = pickle.load(f)
        self.nlp = spacy.load("zh_core_web_sm")

        bin_path = "/data_local/TwoWaysToImproveCSC/large_data/char_pic/character.bin"
        self.img_vector_dict = pickle.load(open(bin_path, "rb"))

        self.model = kenlm.Model("/data_local/slm/chinese_csc.bin")
        print("spell_slm model load success !!")

    def tokenize(self, texts):
        res = []
        for doc in self.nlp.pipe(texts, disable=["tok2vec", "tagger", "parser", "ner"]):
            words = []
            for item in doc:
                words.append(item.text)
            res.append(words)
        return res

    def getscores(self, sents):
        """
        获得语言模型的得分
        :param sents:
        :return:
        """
        res = []
        for text in sents:
            score = self.model.score(text.strip(), bos=False, eos=False)
            res.append(score)
        return res

    def poem_search(self, query_tran='天生我材必有用'):
        """
        匹配诗句，不用分词，至少有5个字一致
        :param query_tran:
        :return:
        """
        query = {
            "query": {
                "match": {
                    "sgl_cont": {
                        "query": query_tran,
                        "minimum_should_match": 5
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
            pattern = ""
            match_character_li = re.findall("<em>(.*?)</em>", hit["highlight"]["sgl_cont"][0])
            if len(hit['_source']['sgl_cont']) - len(match_character_li) <= 4:
                # 与原来的数据差别小于4个字，包括标点
                for w in hit['_source']['sgl_cont']:
                    if w not in match_character_li:
                        pattern += "."
                    else:
                        pattern += w
                str_li.append((pattern, hit['_source']['sgl_cont']))
                # 匹配模式，匹配到的诗句
        return str_li

    def poem_correct(self, text):
        """
        诗句中包含的错误的纠正并返回
        :param text:
        :return:
        """
        res_li = self.poem_search(query_tran=text)
        if len(res_li) == 0:
            return text

        # 已经匹配到的位置，不再处理。
        # 防止将本身正确的改错，偷天换日--移天换日
        correct = {}
        for res in res_li:
            pattern, hit_sgl_cont = res
            if pattern == hit_sgl_cont:
                return text
            else:
                match_idom = re.search(pattern, text)
                if match_idom is not None:
                    s, t = match_idom.span()
                    src = text[s:t]
                    trg = self.poem_post_process(src, hit_sgl_cont)
                    correct[src] = trg
        print(correct)
        for item in correct:
            text = text.replace(item, correct[item])
        return text

    def poem_post_process(self, src, trg):
        """
        如果有拼音相同的 或者 在候选集中的
        :param src:
        :param trgs:
        :return:
        """
        res = ""
        for s, t in zip(list(src), list(trg)):
            if s != t and lazy_pinyin(s) == lazy_pinyin(t):
                res += t
            elif s != t and s in self.confusion_set and t in self.confusion_set[s]:
                res += t
            else:
                res += s
        return res

    def cy_search(self, query_tran='高瞻远瞩'):
        """
        匹配成语，分词，组成四字的匹配
        :param query_tran:
        :return:
        """
        num = len(query_tran)
        query = {
            "query": {
                "match": {
                    "word": {
                        "query": query_tran,
                        "minimum_should_match": num - 1
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
            match_cy = hit['_source']['word']

            if match_cy == query_tran:  # 无错
                return [(query_tran, query_tran)]

            if len(match_cy) != len(query_tran):
                continue

            correct_tag = 0
            count = 0
            for s, t in zip(query_tran, match_cy):
                if s != t and self.is_mix(s, t):
                    correct_tag = 1
                elif s == t:
                    count += 1

            if count >= num - 1 and correct_tag == 1:
                pattern = ""
                match_character_li = re.findall("<em>(.*?)</em>", hit["highlight"]["word"][0])
                for w in match_cy:
                    if w not in match_character_li:
                        pattern += "."
                    else:
                        pattern += w
                str_li.append((pattern, hit['_source']['word']))
                # 匹配模式，匹配到的成语
        return str_li

    def cy_process(self, w_src, src_text):
        """
        查找相似成语
        :param src_text:
        :return:
        """
        res_li = self.cy_search(query_tran=w_src)
        match_trgs = []
        for res in res_li:
            pattern, match_str = res
            if w_src == match_str:
                return w_src
            else:
                match_trgs.append(match_str)
        ##
        if len(match_trgs) == 0:
            return w_src

        candidates = [src_text]
        candidate_words = [w_src]
        for match_trg in match_trgs:
            tmp_text = src_text.replace(w_src, match_trg)
            candidates.append(tmp_text)
            candidate_words.append(match_trg)
        candidate_scores = self.getscores(candidates)
        item = sorted(zip(candidate_words, candidate_scores), key=lambda s: s[1], reverse=True)
        if item[0][1] == item[1][1]:
            return match_trgs[0]
        else:
            return item[0][0]

    def cy_correct(self, text):
        """
        诗句中包含的错误的纠正并返回
        :param text:
        :return:
        """
        words = self.tokenize([text])[0]
        # 前后组成4字候选成语

        correct = {}

        num = len(words)
        for i in range(num - 1):
            if len(words[i]) == 4:
                w_src = words[i]
                start = max(0, i - 2)
                end = min(i + 3, num)
                src_text = " ".join([words[t] for t in range(start, end)])
                w_trg = self.cy_process(w_src, src_text)
                if w_src != w_trg:
                    correct[w_src] = w_trg
            else:
                j = i + 1
                if len(words[i] + words[j]) == 4:
                    w_src = words[i] + words[j]
                    start = max(0, i - 2)
                    end = min(j + 3, num)
                    src_text = " ".join([words[t] for t in range(start, end)])
                    w_trg = self.cy_process(w_src, src_text)
                    if w_src != w_trg:
                        correct[w_src] = w_trg
        print(correct)
        for item in correct:
            text = text.replace(item, correct[item])
        return text

    def cy_post_process(self, src, trgs):
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
                        if s not in self.confusion_set:
                            return src
                        if t in self.confusion_set[s]:
                            print("++在混淆集++", trg)
                            return trg
        return src

    def ci_search(self, query_tran='挫折'):
        """
        找到一个词，对应的候选词
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
        result = self.es.search(index="ci_v4", body=query, size=50)
        hits_list = result['hits']['hits']
        str_li = []
        for hit in hits_list:

            match_str = hit['_source']['ci']
            if query_tran == match_str:
                str_li = [(query_tran, query_tran)]
                return str_li

            if len(match_str) != len(query_tran):
                continue

            correct_tag = 0
            count = 0
            for s, t in zip(query_tran, match_str):
                if s != t and self.is_mix(s, t):
                    correct_tag = 1
                elif s == t:
                    count += 1

            if count >= num - 1 and correct_tag == 1:
                pattern = ""
                match_character_li = re.findall("<em>(.*?)</em>", hit["highlight"]["ci"][0])
                for w in match_str:
                    if w not in match_character_li:
                        pattern += "."
                    else:
                        pattern += w
                str_li.append((pattern, match_str))
                # 匹配模式，匹配到的词语
        return str_li

    def ci_correct_no_lm(self, text):
        """
        :param text:
        :return:
        """
        words = self.tokenize([text])[0]
        # 字数大于1的词语
        four_words = []
        num = len(words)
        for i in range(num - 1):
            if len(words[i]) > 1:
                four_words.append(words[i])

        correct = {}
        for w_src in four_words:
            res_li = self.ci_search(query_tran=w_src)
            word_trgs = []
            for res in res_li:
                pattern, match_word_trg = res
                if w_src == match_word_trg:
                    break
                else:
                    match_idom = re.search(pattern, text)
                    if match_idom is not None:
                        word_trgs.append(match_word_trg)
            if len(word_trgs) >= 1:
                # 用语言模型
                w_cor = self.ci_post_process(w_src, word_trgs)
                correct[w_src] = w_cor
        print(correct)
        for item in correct:
            text = text.replace(item, correct[item])
        return text

    def ci_correct(self, text):
        """
        :param text:
        :return:
        """
        words = self.tokenize([text])[0]
        # 字数大于1的词语
        # four_words = []
        correct = {}
        num = len(words)
        for i in range(num - 1):
            if len(words[i]) > 1:
                w_src = words[i]
                res_li = self.ci_search(query_tran=w_src)
                start = max(0, i - 2)
                end = min(i + 3, num)
                src_text = " ".join([words[t] for t in range(start, end)])
                candidates = [src_text]
                candidate_words = [w_src]
                for res in res_li:
                    pattern, match_word_trg = res
                    if w_src == match_word_trg:
                        break
                    else:
                        match_idom = re.search(pattern, text)
                        if match_idom is not None:
                            tmp_text = src_text.replace(w_src, match_word_trg)
                            candidates.append(tmp_text)
                            candidate_words.append(match_word_trg)

                if len(candidate_words) > 1:
                    # 用语言模型
                    candidate_scores = self.getscores(candidates)
                    item = sorted(zip(candidate_words, candidate_scores), key=lambda s: s[1], reverse=True)
                    if item[0][1] - item[1][1] > -0.3 * item[0][1]:
                        correct[w_src] = item[0][0]
        print(correct)
        for item in correct:
            text = text.replace(item, correct[item])
        return text

    def ci_post_process(self, src, trgs):
        """
        :param src:
        :param trgs:
        :return:
        """
        for trg in trgs:
            for s, t in zip(list(src), list(trg)):
                if s != t:
                    if lazy_pinyin(s) == lazy_pinyin(t) and s in self.confusion_set and t in self.confusion_set[s]:
                        print("++拼音一致++", trg)
                        return trg
                    elif (set(lazy_pinyin(s)[0]) - set(lazy_pinyin(t)[0]) == set(["g"]) or
                          set(lazy_pinyin(t)[0]) - set(lazy_pinyin(s)[0]) == set(["g"])) \
                            and s in self.confusion_set and t in self.confusion_set[s]:
                        print("++前后鼻不分++", trg)
                        return trg
                    # else:
                    #     if s not in confusion_set:
                    #         return src
                    #     if t in confusion_set[s]:
                    #         print("++在混淆集++", trg)
                    #         return trg
        return src

    def is_mix(self, s, t):
        """
        是否考虑多音字
        :param src:
        :param trgs:
        :return:
        """
        s_pinyin = lazy_pinyin(s)[0]
        t_pinyin = lazy_pinyin(t)[0]
        if s_pinyin == t_pinyin:
            print("++拼音一致++")
            return True
        if s in self.confusion_set and t in self.confusion_set[s]:
            print("++在混淆集1++")
            return True
        if t in self.confusion_set and s in self.confusion_set[t]:
            print("++在混淆集2++")
            return True
        if s_pinyin.strip('g') == t_pinyin or t_pinyin.strip('g') == s_pinyin:
            print("++前后鼻不分++")
            return True
        if s_pinyin.replace('zh', 'z') == t_pinyin or t_pinyin.replace('zh', 'z') == s_pinyin:
            print("++平翘舌不分++")
            return True
        if s_pinyin.replace('ch', 'c') == t_pinyin or t_pinyin.replace('ch', 'c') == s_pinyin:
            print("++平翘舌不分++")
            return True
        if s_pinyin.replace('sh', 's') == t_pinyin or t_pinyin.replace('sh', 's') == s_pinyin:
            print("++平翘舌不分++")
            return True

        if s in self.img_vector_dict and t in self.img_vector_dict:

            s_vec = self.img_vector_dict[s]
            t_vec = self.img_vector_dict[t]
            if cosine_similarity(s_vec, t_vec) > 0.96:
                print("++字形相似++")
                return True

        return False


if __name__ == '__main__':
    obj = CscMatch()
    all_texts = [

        # "你的人生方向是什么？你要如何使自己更接近它呢？这只有你自己知道，撰择好一个目标，千万不要轻易放弃它，看好了方向才有努力的于地，有了努力才有成功的机会，好好把握那选择的机会。",
        # "在这小小的空间里，每个人都会分享一些生活点滴，我们也要保持教室的干静，才能安心的延续对这间教室的感情。",
        # "人的方向，由自己来决定。自己的决定都是快乐的，但如果你不能决定你自己的方向，那就要由别人帮你决定，他的决定也许你会不喜欢，但你是要尊从，因为你不能自己决定你人生的方向。",
        # "这件事虽然过了很久，但是当时的心情我却是历历在目，后来我改成不蒸饭了，这种事也不在发生了，当然这件事终就成了回忆，但每当我想起这件事来，真的会不爽骂他个两句呢！",
        # "教室里的黑板每天都帮我们记录回家的功课让我们不怕忘记带东西，墙上的布置像是推动我们支住，班上的争饭机保存著我的食物让我不用担心饿著肚子，情奋的桌子让我们可以安心的写作业，角落的垃圾桶是一个清理垃圾的黑洞，不必单心教室会乱七八糟。",
        # "当我从温暖的被窝爬起来时，外面的战火己停止了，天空也放睛了，再度露出那耀眼的笑容。",
        # "小学时，我常试一个人自己睡觉，那时因为不习惯身边没有人陪著我一起睡觉，所以就一直都睡不觉，而且我还做了一场很可怕的恶梦呢！",
        # "大自然也一样的，无法天天都是晴天，天天都很顺利，但生活，就是如此这般，有失才有得，只是每个人是如何去看侍的。",
        # "曾经听见在考试卷、讲议、功课、课本压得快变肉干的我对自己问：「还想念书吗？还想听课吗？」，但这些在考试与堆积如山的功课贬值成了一杯另人难以咽下的苦水。或许这苦水能让我活耀于分数上，但这是我想要的？」。",
        # "曾经听见在考试卷、讲议、功课、课本压得快变肉干的我对自己问：「还想念书吗？还想听课吗？」，但这些在考试与堆积如山的功课贬值成了一杯另人难以咽下的苦水。或许这苦水能让我活耀于分数上，但这是我想要的？」。"

    ]
    all_trgs = [
        # "你的人生方向是什么？你要如何使自己更接近它呢？这只有你自己知道，撰择好一个目标，千万不要轻易放弃它，看好了方向才有努力的于地，有了努力才有成功的机会，好好把握那选择的机会。",
        # "在这小小的空间里，每个人都会分享一些生活点滴，我们也要保持教室的干静，才能安心的延续对这间教室的感情。",
        # "人的方向，由自己来决定。自己的决定都是快乐的，但如果你不能决定你自己的方向，那就要由别人帮你决定，他的决定也许你会不喜欢，但你是要尊从，因为你不能自己决定你人生的方向。",
        # "这件事虽然过了很久，但是当时的心情我却是历历在目，后来我改成不蒸饭了，这种事也不在发生了，当然这件事终就成了回忆，但每当我想起这件事来，真的会不爽骂他个两句呢！",
        # "教室里的黑板每天都帮我们记录回家的功课让我们不怕忘记带东西，墙上的布置像是推动我们支住，班上的争饭机保存著我的食物让我不用担心饿著肚子，情奋的桌子让我们可以安心的写作业，角落的垃圾桶是一个清理垃圾的黑洞，不必单心教室会乱七八糟。",
        # "当我从温暖的被窝爬起来时，外面的战火己停止了，天空也放睛了，再度露出那耀眼的笑容。",
        # "小学时，我常试一个人自己睡觉，那时因为不习惯身边没有人陪著我一起睡觉，所以就一直都睡不觉，而且我还做了一场很可怕的恶梦呢！",
        # "大自然也一样的，无法天天都是晴天，天天都很顺利，但生活，就是如此这般，有失才有得，只是每个人是如何去看侍的。",
        # "曾经听见在考试卷、讲议、功课、课本压得快变肉干的我对自己问：「还想念书吗？还想听课吗？」，但这些在考试与堆积如山的功课贬值成了一杯另人难以咽下的苦水。或许这苦水能让我活耀于分数上，但这是我想要的？」。",
        # "曾经听见在考试卷、讲议、功课、课本压得快变肉干的我对自己问：「还想念书吗？还想听课吗？」，但这些在考试与堆积如山的功课贬值成了一杯另人难以咽下的苦水。或许这苦水能让我活耀于分数上，但这是我想要的？」。"

    ]
    paths = [
        "/data_local/TwoWaysToImproveCSC/BERT/data/13test.txt",
        "/data_local/TwoWaysToImproveCSC/BERT/cc_data/chinese_spell_4.txt"
    ]
    for path in paths:
        all_texts = []
        all_trgs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                sent, trg = line.strip().split(" ")
                all_texts.append(sent)
                all_trgs.append(trg)

        path_out = path.replace(".txt", ".pre")

        with open(path_out, "w", encoding="utf-8") as fw:
            num = len(all_texts)
            for i in range(num):
                print("句子：", str(i))
                text = all_texts[i]
                trg = all_trgs[i]
                # text = obj.cy_correct(text)
                # text = obj.ci_correct(text)
                new_text = obj.poem_correct(text)

                fw.write(text + " " + new_text + "\n")

                if text != new_text:
                    item_pre = set()
                    for s, t in zip(text, new_text):
                        if s != t:
                            item_pre.add((s, t))

                    item_label = set()
                    for s, t in zip(text, trg):
                        if s != t:
                            item_label.add((s, t))

                    if len(item_pre - item_label) != 0:
                        print("============")
                        print(text)
                        print(new_text)
                        print(trg)

    # with open("./error_ci.txt", "w", encoding="utf-8") as fw:
    #     num = len(all_texts)
    #     for i in range(num):
    #         print("句子：", str(i))
    #         text = all_texts[i]
    #         trg = all_trgs[i]
    #         new_text = obj.ci_correct(text)
    #         if text != new_text:
    #             item_pre = set()
    #             for s, t in zip(text, new_text):
    #                 if s != t:
    #                     item_pre.add((s, t))
    #
    #             item_label = set()
    #             for s, t in zip(text, trg):
    #                 if s != t:
    #                     item_label.add((s, t))
    #
    #             if len(item_pre - item_label) != 0:
    #                 # print("============")
    #                 fw.write(text + "\n")
    #                 print(text)
    #                 print(new_text)
    #                 print(trg)
    #
    #     for text in all_texts:
    #         new_text = obj.cy_correct(text)
    #         if text != new_text:
    #             print(text)
    #             print(new_text)

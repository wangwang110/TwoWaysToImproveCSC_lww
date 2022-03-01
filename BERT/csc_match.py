# -*- coding: utf-8 -*-

from elasticsearch import Elasticsearch
import kenlm
import re
import os
import jieba
import pickle
from pypinyin import pinyin, lazy_pinyin, Style

"""
定义一些拼音错误的规则
1.前后鼻音不分
2.平翘舌部分
3. 字符图片相似度
"""


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


class CSCmatch:
    def __init__(self):
        self.es = Elasticsearch([{"host": "10.21.2.35", "port": 9201}])
        # 候选集
        filepath = '/data_local/TwoWaysToImproveCSC/BERT/save/confusion.file'
        with open(filepath, 'rb') as f:
            self.confusion_set = pickle.load(f)

        # 增加的混淆集
        add_dict = {"辙": ["彻"]}
        for s in add_dict:
            if s not in self.confusion_set:
                self.confusion_set[s] = set()
            for w in add_dict[s]:
                self.confusion_set[s].add(w)

        # 词语先不做，容易引入错误
        self.ci_dict = pickle.load(open("/data_local/TwoWaysToImproveCSC/BERT/chinese-xinhua/match_ci.pkl", "rb"))
        self.ci_set = pickle.load(open("/data_local/TwoWaysToImproveCSC/BERT/chinese-xinhua/ci_set.pkl", "rb")) \
                      | pickle.load(open("/data_local/TwoWaysToImproveCSC/BERT/chinese-xinhua/ci_set_spacy.pkl", "rb"))

        # 成语匹配字典
        self.cy_dict = pickle.load(open("/data_local/TwoWaysToImproveCSC/BERT/chinese-xinhua/match_cy.pkl", "rb"))
        self.cy_set = pickle.load(open("/data_local/TwoWaysToImproveCSC/BERT/chinese-xinhua/cy_set.pkl", "rb"))

        # bin_path = "/data_local/TwoWaysToImproveCSC/large_data/char_pic/character.bin"
        # self.img_vector_dict = pickle.load(open(bin_path, "rb"))

        self.model = kenlm.Model("/data_local/slm/chinese_csc_char.bin")
        print("model load success !!")

    def tokenize(self, texts):
        """
        分词 jieba.cut("我来到北京清华大学", cut_all=False)
        :param texts:
        :return:
        """
        res = []
        for text in texts:
            cut_gen = jieba.cut(text, cut_all=False)
            res.append(list(cut_gen))
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
        匹配诗句，不用分词，至少有6个字一致
        不知道es怎么实现这个6个字必须是连续的，只能先匹配再用规则
        :param query_tran:
        :return:
        """
        query = {
            "query": {
                "match": {
                    "sgl_cont": {
                        "query": query_tran,
                        "minimum_should_match": 6
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
        # 默认一个输入最多只包含1个诗句
        hits_list = result['hits']['hits']
        str_li = []
        for hit in hits_list:
            match_str = hit['_source']['sgl_cont']
            match_character_li = re.findall("<em>(.*?)</em>", hit["highlight"]["sgl_cont"][0])
            if len(match_str) - len(match_character_li) <= 4:
                # 与原来的数据差别小于4个字，包括标点
                pattern_li = [w if w in match_character_li else "." for w in match_str]
                match_idom = re.search("".join(pattern_li), query_tran)
                if match_idom is not None:
                    low, high = match_idom.span()
                    src_text = query_tran[low:high]
                    if match_str == src_text:
                        res = (low, high, src_text, match_str)
                        str_li.append(res)
                    else:
                        trg_tokens = []
                        for s, t in zip(list(src_text), list(match_str)):
                            if s != t and self.is_mix(s, t):
                                trg_tokens.append(t)
                            else:
                                trg_tokens.append(s)
                        res = (low, high, src_text, "".join(trg_tokens))
                        str_li.append(res)
        return str_li

    def poem_correct(self, texts):
        """
        诗句中包含的错误的纠正并返回
        :param text:
        :return:
        """
        sents_res_li = []
        res_poem_li = []
        for text in texts:
            query_text = text
            change_poem_li = []
            res_li = self.poem_search(query_tran=query_text)
            if len(res_li) == 0:
                sents_res_li.append(query_text)
                res_poem_li.append([])
                continue

            for res in res_li:
                i, j, src_text, trg_text = res
                change_poem_li.extend([p for p in range(i, j)])
                query_text = query_text.replace(src_text, trg_text)
            sents_res_li.append(query_text)
            res_poem_li.append([])
        return sents_res_li, res_poem_li

    def cy_candidates(self, query_tran='高瞻远瞩'):
        """
        匹配成语
        :param query_tran:
        :return:
        """
        candidates = set()
        num = len(query_tran)
        for i in range(num):
            tokens = list(query_tran)
            tokens[i] = "_"
            key = "".join(tokens)
            if key in self.cy_dict:
                candidates = candidates | self.cy_dict[key]

        #
        final_res = []
        for match_str in candidates:
            for s, t in zip(query_tran, match_str):
                if s != t:
                    if self.is_mix(s, t):
                        final_res.append(match_str)
                    break
        return final_res

    def cy_correct(self, texts, change_li_all):
        """
        :param text:
        :param no_change_li:
        :return:
        """
        res_li = []
        # res_poem_li = []

        for text, change_li in zip(texts, change_li_all):
            query_text = text
            words = self.tokenize([text])[0]
            # 相邻两个组成4字候选成语
            # jieba分词？
            s = 0
            correct = {}
            cy_li = []
            num = len(words)
            for i in range(num):
                for d in range(1, 5, 1):
                    j = i + d
                    if j < num:
                        word_str = "".join(words[i:j])
                        if len(word_str) == 4 and word_str not in self.cy_set \
                                and not self.is_no_change(s, word_str, change_li):
                            w_src = word_str
                            match_trgs = self.cy_candidates(w_src)
                            if len(match_trgs) != 0:
                                w_trg = self.cy_lm_correct_part(match_trgs, i, j, words)
                                if w_src != w_trg:
                                    correct[w_src] = w_trg
                                    cy_li.extend([s + p for p in range(len(word_str))])
                s += len(words[i])
            # print(correct)
            for item in correct:
                query_text = query_text.replace(item, correct[item])
            res_li.append(query_text)
        return res_li

    def ci_candidates(self, query_tran='高瞻远瞩'):
        """
        匹配成语
        :param query_tran:
        :return:
        """
        candidates = set()
        num = len(query_tran)
        for i in range(num):
            tokens = list(query_tran)
            tokens[i] = "_"
            key = "".join(tokens)
            if key in self.ci_dict:
                candidates = candidates | self.ci_dict[key]
        #
        final_res = []
        for match_str in candidates:
            for s, t in zip(query_tran, match_str):
                if s != t:
                    if self.is_mix(s, t):
                        final_res.append(match_str)
                    break
        return final_res

    def is_tokenize_false(self, words, i):
        """
        判断分词有没有错，与前后字能否组成词
        :param words:
        :return:
        """
        low = len("".join(words[:i])) + 1
        high = len("".join(words[:i + 1])) - 1
        tokens = list("".join(words))
        for p in range(2, 4):
            s = max(0, low - p)
            t = min(high + p, len(tokens))
            if "".join(tokens[s:low]) in self.ci_set:
                return False
            if "".join(tokens[high:t]) in self.ci_set:
                return False
        return True

    def ci_correct(self, text, change_li):
        """

        :param text:
        :param change_li:
        :return:
        """
        query_text = text
        words = self.tokenize([text])[0]
        # 相邻两个组成4字候选成语
        s = 0
        correct = {}
        ci_li = []
        num = len(words)
        for i in range(num - 1):
            if 1 < len(words[i]) < 4 and words[i] not in self.ci_set \
                    and not self.is_no_change(s, words[i], change_li):
                w_src = words[i]
                match_trgs = self.ci_candidates(w_src)
                if len(match_trgs) != 0 and w_src not in match_trgs:
                    w_trg = self.cy_lm_correct_part(match_trgs, i, i + 1, words)
                    if w_src != w_trg and self.is_tokenize_false(words, i):
                        correct[w_src] = w_trg
                        ci_li.extend([s + p for p in range(len(w_src))])
        # print(correct)
        for item in correct:
            query_text = query_text.replace(item, correct[item])
        return query_text, ci_li

    def is_mix(self, s, t):
        """
        是否考虑多音字
        :param src:
        :param trgs:
        :return:
        """
        # s_pinyin = lazy_pinyin(s)[0]
        # t_pinyin = lazy_pinyin(t)[0]

        s_pinyin = pinyin(s)[0][0]
        t_pinyin = pinyin(t)[0][0]
        if s_pinyin == t_pinyin:
            # print("++拼音一致++")
            return True
        if s in self.confusion_set and t in self.confusion_set[s]:
            # print("++在混淆集1++")
            return True
        if t in self.confusion_set and s in self.confusion_set[t]:
            # print("++在混淆集2++")
            return True
        if s_pinyin.strip('g') == t_pinyin or t_pinyin.strip('g') == s_pinyin:
            # print("++前后鼻不分++")
            return True
        if s_pinyin.replace('zh', 'z') == t_pinyin or t_pinyin.replace('zh', 'z') == s_pinyin:
            # print("++平翘舌不分++")
            return True
        if s_pinyin.replace('ch', 'c') == t_pinyin or t_pinyin.replace('ch', 'c') == s_pinyin:
            # print("++平翘舌不分++")
            return True
        if s_pinyin.replace('sh', 's') == t_pinyin or t_pinyin.replace('sh', 's') == s_pinyin:
            # print("++平翘舌不分++")
            return True
        # 计算很慢
        # 字形相似度（图片向量）
        # if s in self.img_vector_dict and t in self.img_vector_dict:
        #     s_vec = self.img_vector_dict[s]
        #     t_vec = self.img_vector_dict[t]
        #     if cosine_similarity(s_vec, t_vec) > 0.96:
        #         print("++字形相似++")
        #         return True

        return False

    def cy_lm_correct_part(self, match_trgs, start, end, words, name="cy"):
        """
        查找相似成语
        :param src_text:
        :return:
        """
        # 前后个五个字
        # A B C D EF AB  C D E F G
        # 0 1 2 3  4  5  6 7 8 9 10

        # A B C D E F A B C D E  F   G
        # 0 1 2 3 4 5 6 7 8 9 10 11 12

        w_src_str = "".join(words[start:end])
        w_src = " ".join(list(w_src_str))

        src_sent = "".join(words)

        low = max(0, len("".join(words[:start])) - 4)  # 4  # start - 4
        high = min(len(src_sent), len("".join(words[:end])) + 4)  # 8  #  end + 4

        src_text = " ".join(list(src_sent)[low:high])
        #
        # print(w_src)
        # print(src_text)

        candidates = [src_text]
        candidate_words = [w_src_str]
        for match_trg in match_trgs:
            match_str = " ".join(list(match_trg))
            tmp_text = src_text.replace(w_src, match_str)
            candidates.append(tmp_text)
            candidate_words.append(match_trg)
        candidate_scores = self.getscores(candidates)
        item = sorted(zip(candidate_words, candidate_scores), key=lambda s: s[1], reverse=True)
        # print(w_src)
        # print(item)
        if name == 'cy':
            return item[0][0]
        elif name == 'ci':
            if item[0][1] - item[1][1] > -0.3 * item[0][1]:
                return item[0][0]
            else:
                return w_src

    def is_no_change(self, start, word, change_li):
        """
        诗词修改后的成语不修改
        成语修改后的词语不修改
        :param start:
        :param word:
        :param no_change_li:
        :return:
        """
        no_change_tag = 0
        for step in range(len(word)):
            if start + step in change_li:
                no_change_tag = 1
                break
        return no_change_tag


if __name__ == '__main__':
    obj = CSCmatch()
    all_texts = [
        "人生中一定有不顺的事，但不能被打败，要坚持自己的意念，不管别人言语、行动去干扰你，你也能继续的往前走，不管多么多的拌脚石，你也能一次一次的站起来，把握时间，要做的事就去做，不犹豫，免得时机错过后悔莫及。",
        "刘墉在三岁过年时，全家陷入火海，把家烧得面目全飞、体无完肤。"
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
        "人生中一定有不顺的事，但不能被打败，要坚持自己的意念，不管别人言语、行动去干扰你，你也能继续的往前走，不管多么多的拌脚石，你也能一次一次的站起来，把握时间，要做的事就去做，不犹豫，免得时机错过后悔莫及。",
        "刘墉在三岁过年时，全家陷入火海，把家烧得面目全飞、体无完肤。"
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
        "/data_local/TwoWaysToImproveCSC/BERT/chinese-xinhua/bert_out_cc.txt"
        # "./13test_tmp.txt",
        # "/data_local/TwoWaysToImproveCSC/BERT/data/13test.txt",
        # "/data_local/TwoWaysToImproveCSC/BERT/cc_data/chinese_spell_4.txt"
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
                print("\n\n\n句子：", str(i))
                text = all_texts[i]
                trg = all_trgs[i]
                change_pos = []

                # 诗词匹配
                poem_text, poem_li = obj.poem_correct(text)
                change_pos.extend(poem_li)

                # 成语匹配
                cy_text, cy_li = obj.cy_correct(poem_text, change_pos)
                change_pos.extend(cy_li)

                # 词语匹配，容易引入错误，包括分词不准，专有名词，造成的错误
                # 即使在模型的输出之后做也会有同样的问题
                # 如果并行合并，召回率又会低很多
                # ci_text = cy_text
                ci_text, ci_li = obj.ci_correct(cy_text, change_pos)

                fw.write(text + " " + ci_text + "\n")

                if text != ci_text:
                    item_pre = set()
                    for s, t in zip(text, ci_text):
                        if s != t:
                            item_pre.add((s, t))

                    item_label = set()
                    for s, t in zip(text, trg):
                        if s != t:
                            item_label.add((s, t))

                    if len(item_pre - item_label) != 0:
                        print("============")
                        print(text)
                        print(ci_text)
                        print(trg)

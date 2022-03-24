# -*- coding: UTF-8 -*-

import sys
import os
import re
from log.logger import logger_fn
from csc_utils import cut_sent, replace_space, remove_space
from csc_model import CSCmodel
from csc_match import CSCmatch

base_path = os.path.realpath(__file__)
base_path = base_path[:base_path.rfind("/")]
sys.path.append(base_path)

logger = logger_fn('csc_main', base_path + '/log/csc_main.log')
import string


class CSC:
    def __init__(self, bert_path, model_path):
        """
        :param bert_path:
        :param model_path:
        :param gpu_id:
        """
        self.model = CSCmodel(bert_path, model_path)
        self.match_model = CSCmatch()
        self.match_process_tag = 0
        # self.punct = ["…", "‘", "’", "“", "”","：","；"]
        self.punct = string.punctuation

    def correct(self, data):
        """
        错别字纠正
        1. 返回输入文本，输出文本，修改列表 == 哪个位置 == 原来是哪个字 ==替换成哪个字

        模型前后要做哪些后处理，

        句子输入，文章输入要做哪些前处理 （借鉴语法纠错）

        2. 要不要返回检索到的名人名言，诗词，成语


        :param data:
        :return:
        """

        output_dict = {
            "status": 0,
            "msg": u"错别字纠正完成",
            "data": {"input_text": "", "output_text": "", "edits": [], "chengyu": [], "poem": [], "mrmy": []}}

        try:
            original_text = str(data.get("article"))
            logger.info(u"用户输入文本：{}".format(original_text))
        except:
            output_dict["status"] = 10
            output_dict["msg"] = u"输入格式有误"
            logger.error(u"输入格式有误", exc_info=True)
            return output_dict

        if len(original_text.strip()) > 2000:
            output_dict["msg"] = u"输入文本字符个数超过2000"
            output_dict["status"] = 11
            return output_dict

        letters = re.sub('[^\u4e00-\u9fa5]', "", original_text.strip())
        if len(letters) / len(original_text) <= 0.25:
            output_dict["msg"] = u"输入文本非中文字符较多"
            output_dict["status"] = 12
            return output_dict

        sent_li = cut_sent(original_text)
        # 不影响位置
        input_li_space = []
        input_li = []
        for sent in sent_li:
            input_li_space.append(replace_space(sent))
            input_li.append(remove_space(sent))

        _, position_mapping = self.get_mapping_positon("".join(input_li_space), "".join(input_li))

        if len("".join(input_li_space)) != len(original_text):
            output_dict["msg"] = u"系统错误"
            output_dict["status"] = 13
        else:
            all_pos = [[] for _ in range(len(input_li))]
            if self.match_process_tag:
                src_li, output_mymy_li, output_poem_li, output_cy_li = self.match_sents(input_li, all_pos)
                output_dict["data"]["mrmy"] = output_mymy_li
                output_dict["data"]["poem"] = output_poem_li
                output_dict["data"]["chengyu"] = output_cy_li
            else:
                src_li = input_li

            trg_li = self.model.test(src_li)
            reset_trg_li = self.post_process(src_li, trg_li, all_pos)

            # 获得修改列表，位置对应好
            edits = []
            pos = 0
            for src, trg in zip(input_li, reset_trg_li):
                # src = src.replace("[UNK]", "#")
                trg = trg.replace("[UNK]", "#")
                for s, t in zip(src, trg):
                    if s in self.punct or t == "#":
                        # 不做修改的
                        pos += 1
                        # 每次位置是要变化的，[UNK]占了五个字符
                        # 有些单词love被认为是一个词，但是修改之后会替换回来
                        continue
                    if s != t:
                        edit = {"pos": position_mapping[pos], "src_token": s, "trg_token": t}
                        edits.append(edit)
                    pos += 1

            output_dict["data"]["input_text"] = original_text
            tokens = list(original_text)
            for e in edits:
                pos = e["pos"]
                src_token = e["src_token"]
                trg_token = e["trg_token"]
                if tokens[pos] == src_token:
                    tokens[pos] = trg_token

            output_dict["data"]["output_text"] = "".join(tokens)
            output_dict["data"]["edits"] = edits

        return output_dict

    def post_process(self, src_li, trg_li, all_pos):
        """
        前面已经匹配纠正的不做处理
        :param src_li:
        :param trg_li:
        :param all_pos:
        :return:
        """
        final_res_li = []
        for src_sent, trg_sent, pos_li in zip(src_li, trg_li, all_pos):
            pos = 0
            src_tokens = list(src_sent)
            trg_tokens = list(trg_sent)
            for s, t in zip(src_tokens, trg_tokens):
                if s != t and pos in pos_li:
                    trg_tokens[pos] = src_tokens[pos]
                pos += 1
            final_res_li.append("".join(trg_tokens))
        return final_res_li

    def help_match_sents(self, input_li, match_poem_li, pos_poem_li, all_pos, name):
        """
        根据修正返回原句
        :param input_li:
        :param match_poem_li:
        :return:
        """
        match_res_li = []
        res_li = []
        num = len(input_li)
        for i in range(num):
            query_text = input_li[i]
            for src_text in match_poem_li[i]:
                if name == "poem":
                    trg_text, info = match_poem_li[i][src_text].split("####")
                else:
                    trg_text = match_poem_li[i][src_text]
                    info = ""
                query_text = query_text.replace(src_text, trg_text)
                if src_text == trg_text:
                    match_item = {"src": src_text, "trg": trg_text, "info": info}
                else:
                    match_item = {"src": src_text, "trg": trg_text, "info": info}
                match_res_li.append(match_item)
            res_li.append(query_text)

        for pos_li, poem_li in zip(all_pos, pos_poem_li):
            pos_li.extend(poem_li)

        return res_li, match_res_li

    def match_sents(self, input_li, all_pos):
        """
        匹配名人名言，诗词，成语，并返回结果
        :return:
        """

        # 1. 匹配名人名言, 不纠正
        match_mrmy_li = self.match_model.mrmy_correct(input_li)
        output_mymy_li = []
        for i in range(len(input_li)):
            for src_text in match_mrmy_li[i]:
                author = match_mrmy_li[i][src_text]
                match_item = {"src": input_li[i], "trg": src_text, "info": author}
                output_mymy_li.append(match_item)

        # 2. 匹配诗句，纠正
        match_poem_li, pos_poem_li = self.match_model.poem_correct(input_li)
        res_poem_li, output_poem_li = self.help_match_sents(input_li, match_poem_li, pos_poem_li, all_pos, name="poem")

        # 3. 匹配成语
        match_cy_li, pos_cy_li = self.match_model.cy_correct(res_poem_li, all_pos)
        res_cy_li, output_cy_li = self.help_match_sents(res_poem_li, match_cy_li, pos_cy_li, all_pos, name="chengyu")

        return res_cy_li, output_mymy_li, output_poem_li, output_cy_li

    def get_mapping_positon(self, source, candidate):
        i = 0
        j = 0
        new_edits = []
        dict_pos_map = {}
        while i < len(source) and j < len(candidate):
            if source[i] == candidate[j]:
                new_edits.append((i, j))
                dict_pos_map[j] = i
                i += 1
                j += 1
            elif source[i] == " ":
                new_edits.append((i, -1))
                i += 1
            elif candidate[j] == " ":
                new_edits.append((-1, j))
                dict_pos_map[j] = -1
                j += 1
            else:
                i += 1
                j += 1
        if i < len(source):
            new_edits.append((i, -1))
        if j < len(candidate):
            new_edits.append((-1, j))
            dict_pos_map[j] = -1
        return new_edits, dict_pos_map


if __name__ == "__main__":
    # 初始化模型
    bert_path = "/data_local/plm_models/chinese_L-12_H-768_A-12/"
    load_path = "/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_mask/sighan13/model.pkl"
    obj = CSC(bert_path, load_path)
    texts = [
        "……在世界上，没有一条路是可以让人安全走过的，每一条路都有其崎岖之处在等待人们掉入，这些险阱便是痛苦、失望、难过、挫折……",
        "风和日丽",
        "爱迪生曾说过：天才是百分之九十九的汗水，加百分之一的天分",
        "时间就是生命。赶快行动起来吧！",
        "剪不断，理还乱，是离愁，别是一办滋味在心头。四年级:李孜",
        "我见过一望无际、波澜壮阔的大海；玩赏过水平如镜、诗情画意的西湖；游览过翡翠般的漓江；让我难以忘怀的要数那荷叶飘香、群山坏绕的普者黑。",
        "「不禁一番寒辙骨，焉得梅花扑鼻香。」这就是最好的说明，没有失败哪里来的成功呢？",
        "今天天气很好 ，风和日里，出去玩吧，你 觉得如喝。这个月冲值有优惠吗？我这个月重置了话费？请帮我查木月的流量；",
        "我们要牢记见贤思其，见不贤而内自醒",
        "我爱北进天安门。。。我爱北京天按门",
        "我爱北京天按门",
        "没过几分钟，救护车来了，发出  响亮而清翠的声音",
    ]
    for text in texts:
        print(text)
        data = {
            "article": text
        }
        res = obj.correct(data)
        print(res)

        for item in res['data']['edits']:
            print(item["src_token"] + "==" + list(text)[item["pos"]])
            print(item["trg_token"])

    with open("./data/13test_lower.txt", "r", encoding="utf-8")as f, \
            open("./data/13test_lower_model.txt", "w", encoding="utf-8") as fw:
        for line in f.readlines():
            src, trg = line.strip().split()
            data = {
                "article": src
            }
            res = obj.correct(data)
            fw.write(src + " " + res["data"]["output_text"] + "\n")

    with open("./cc_data/chinese_spell_lower_4.txt", "r", encoding="utf-8")as f, \
            open("./data/chinese_spell_lower_4_model.txt", "w", encoding="utf-8") as fw:
        for line in f.readlines():
            src, trg = line.strip().split()
            data = {
                "article": src
            }
            res = obj.correct(data)
            fw.write(src + " " + res["data"]["output_text"] + "\n")

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


class CSC:
    def __init__(self, bert_path, model_path):
        """
        :param bert_path:
        :param model_path:
        :param gpu_id:
        """
        self.model = CSCmodel(bert_path, model_path)
        self.match_model = CSCmatch()
        self.match_process_tag = 1
        self.punct = ["…", "‘", "’", "“", "”"]

    def correct(self, data):
        """
        错别字纠正
        1. 返回输入句子，输出句子，修改列表 == 哪个位置 == 原来是哪个字 ==替换成哪个字

        模型前后要做哪些后处理，

        句子输入，文章输入要做哪些前处理 （借鉴语法纠错）

        :param data:
        :return:
        """

        output_dict = {
            "status": 0,
            "msg": u"错别字纠正完成",
            "data": {"input_text": "", "output_text": "", "edits": []}}

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
            print('=============================== error ===============================')
        else:
            if self.match_process_tag:
                res_li, res_poem_li = self.match_model.poem_correct(input_li)
                src_li = self.match_model.cy_correct(res_li, res_poem_li)
            else:
                src_li = input_li

            _, trg_li = self.model.test(src_li)

            # 获得修改列表，位置对应好
            edits = []
            pos = 0
            new_trg_li = []
            for src, trg in zip(input_li, trg_li):
                new_trg = []
                for s, t in zip(src, trg):
                    if s in self.punct:
                        # 不做修改的
                        new_trg.append(s)
                        pos += 1
                        continue

                    if s != t:
                        edit = {"pos": position_mapping[pos], "src_token": s, "trg_token": t}
                        edits.append(edit)
                    new_trg.append(s)
                    pos += 1
                new_trg_li.append("".join(new_trg))

            output_dict["data"]["input_text"] = "".join(input_li)
            output_dict["data"]["output_text"] = "".join(trg_li)
            output_dict["data"]["edits"] = edits

        return output_dict

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
    load_path = "/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998/sighan13/model.pkl"
    obj = CSC(bert_path, load_path)
    texts = [
        "今天天气很好 ，风和日里，出去玩吧，你 觉得如喝。这个月冲值有优惠吗？我这个月重置了话费？请帮我查木月的流量；",
        "我们要牢记见贤思其，见不贤而内自醒",
        "我爱北进天安门。。。我爱北京天按门",
        "我爱北京天按门",
        "没过几分钟，救护车来了，发出响亮而清翠的声音",
        "我见过一望无际、波澜壮阔的大海；玩赏过水平如镜、诗情画意的西湖；游览过翡翠般的漓江；让我难以忘怀的要数那荷叶飘香、群山坏绕的普者黑。"
    ]
    for text in texts:
        print(text)
        data = {
            "article": text
        }
        res = obj.correct(data)
        # print(res)

        for item in res['data']['edits']:
            print(item["src_token"] + "==" + list(text)[item["pos"]])
            print(item["trg_token"])

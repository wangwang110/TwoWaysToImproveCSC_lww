# -*- coding: utf-8 -*-

if __name__ == '__main__':
    paths = [
        # "/data_local/TwoWaysToImproveCSC/large_data/tmp.txt"
        "/data_local/TwoWaysToImproveCSC/large_data/new2016zh/news2016zh_cor.txt",
        "/data_local/TwoWaysToImproveCSC/large_data/webtext2019zh/web_zh_correct.txt",
        "/data_local/TwoWaysToImproveCSC/large_data/translation2019zh/translation2019zh_correct.txt",
        "/data_local/TwoWaysToImproveCSC/BERT/cc_data/xiaoxue_sent_all_cor.txt",
        "/data_local/TwoWaysToImproveCSC/large_data/weibo/weibo_correct_first.txt",
        "/data_local/TwoWaysToImproveCSC/large_data/weibo/weibo_correct_second.txt",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_00_sent",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_01_sent",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_02_sent",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_03_sent",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_04_sent",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_05_sent",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_06_sent",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_07_sent",
        "/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_sent/wiki_08_sent",
    ]

    path_out = "/data_local/slm/chinese_data.char"
    with open(path_out, "w", encoding="utf-8") as fw:
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    tokens = list(line.strip())
                    fw.write(" ".join(tokens) + "\n")

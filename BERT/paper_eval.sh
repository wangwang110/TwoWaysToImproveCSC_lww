#!/usr/bin/env bash
# 使用论文公开的模型进行测试，分析



#baseline="/data_local/TwoWaysToImproveCSC/BERT/save/bert_paper_model/baseline/sighan13/model.pkl"
preTrain="/data_local/TwoWaysToImproveCSC/BERT/save/bert_paper_model/preTrain/sighan13/model.pkl"
#advTrain="/data_local/TwoWaysToImproveCSC/BERT/save/bert_paper_model/advTrain/sighan13/adv.pkl"
gpu=6

# sighan13
# CUDA_VISIBLE_DEVICES=$gpu python bft_eval.py  --task_name=paper --gpu_num=1 --load_model=True  --load_path=$baseline --do_test=True --test_data="./data/13test.txt" --batch_size=16


CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=paper --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data="./data/13test.txt" --batch_size=16


#CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=paper --gpu_num=1 --load_model=True  --load_path=$advTrain --do_test=True --test_data="./data/13test.txt" --batch_size=16


# xaioxue
#CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=paper --gpu_num=1 --load_model=True  --load_path=$baseline --do_test=True --test_data="./cc_data/xiaoxue_tc_4_sent.txt" --batch_size=16


CUDA_VISIBLE_DEVICES=$gpu  python bft_eval_no_trg.py --task_name=paper --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data="./cc_data/xiaoxue_tc_4_sent.txt" --batch_size=16


#CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=paper --gpu_num=1 --load_model=True  --load_path=$advTrain --do_test=True --test_data="./cc_data/xiaoxue_tc_4_sent.txt" --batch_size=16
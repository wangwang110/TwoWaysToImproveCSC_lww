#!/usr/bin/env bash
# 使用论文公开的模型进行测试，分析
# 作文测试集，可以先只测一个结果（后面决定适配领域的时候再做处理）


preTrain="/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/sighan13/model.pkl"
gpu=5

# sighan13
# CUDA_VISIBLE_DEVICES=$gpu python bft_eval_no_trg.py  --task_name=paper --gpu_num=1 --load_model=True  --load_path=$baseline --do_test=True --test_data="./data/13test.txt" --batch_size=16


CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=self --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data="./data/13test.txt" --batch_size=16


# CUDA_VISIBLE_DEVICES=$gpu  python bft_eval_no_trg.py --task_name=paper --gpu_num=1 --load_model=True  --load_path=$advTrain --do_test=True --test_data="./data/13test.txt" --batch_size=16


## xaioxue
#CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=paper --gpu_num=1 --load_model=True  --load_path=$baseline --do_test=True --test_data="./cc_data/xiaoxue_tc_4_sent.txt" --batch_size=16
#
#
CUDA_VISIBLE_DEVICES=$gpu  python bft_eval_no_trg.py --task_name=self --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data="./cc_data/zuowen_4.test" --batch_size=16
#
#
#CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=paper --gpu_num=1 --load_model=True  --load_path=$advTrain --do_test=True --test_data="./cc_data/xiaoxue_tc_4_sent.txt" --batch_size=16
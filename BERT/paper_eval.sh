#!/usr/bin/env bash
# 使用论文公开的模型进行测试，分析



#baseline="/data_local/TwoWaysToImproveCSC/BERT/save/bert_paper_model/baseline/sighan13/model.pkl"
preTrain="/data_local/TwoWaysToImproveCSC/BERT/save/bert_paper_model/preTrain/sighan13/model.pkl"
#advTrain="/data_local/TwoWaysToImproveCSC/BERT/save/bert_paper_model/advTrain/sighan13/adv.pkl"


gpu=3

data=./data/13test.txt
data_lower=./data/13test_lower.txt
data_pre=./data/13test_cy_new.txt
task=paper

# sighan13
#CUDA_VISIBLE_DEVICES=$gpu python bft_eval.py  --task_name=paper --gpu_num=1 --load_model=True  --load_path=$baseline #--do_test=True --test_data="./data/13test.txt" --batch_size=16

CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data=$data --batch_size=16

CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data=$data_pre --batch_size=16


#CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=paper --gpu_num=1 --load_model=True  --load_path=$advTrain #--do_test=True --test_data="./data/13test.txt" --batch_size=16


data=./cc_data/chinese_spell_4.txt
data_lower=./cc_data/chinese_spell_lower_4.txt
data_pre=./cc_data/chinese_spell_cy_new_4.txt

# xaioxue
#CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=paper --gpu_num=1 --load_model=True  --load_path=$baseline #--do_test=True --test_data="./data_analysis/chinese_spell_4.txt" --batch_size=16


CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data=$data --batch_size=16

CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data=$data_pre --batch_size=16

#CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=paper --gpu_num=1 --load_model=True  --load_path=$advTrain #--do_test=True --test_data="./data_analysis/chinese_spell_4.txt" --batch_size=16



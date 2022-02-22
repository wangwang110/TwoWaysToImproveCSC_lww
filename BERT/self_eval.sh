#!/usr/bin/env bash
# 使用论文公开的模型进行测试，分析



#baseline="/data_local/TwoWaysToImproveCSC/BERT/save/baseline/sighan13/model.pkl"
#preTrain="/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/sighan13_no_ner/model.pkl"
#nerTrain="/data_local/TwoWaysToImproveCSC/BERT/save/nertrain/sighan13/model.pkl"


baseline="/data_local/TwoWaysToImproveCSC/BERT/save/test/wiki_00_lower/sighan13/model.pkl"
preTrain="/data_local/TwoWaysToImproveCSC/BERT/save/test/wiki_00_uniform/sighan13/model.pkl"
nerTrain="/data_local/TwoWaysToImproveCSC/BERT/save/test/wiki_00_lower_confuse/sighan13/model.pkl"

gpu=7

data=./data/13test_lower.txt
data1=./data/13test_uniform.txt
task=test

# sighan13
echo $baseline
CUDA_VISIBLE_DEVICES=$gpu python bft_eval.py  --task_name=$task --gpu_num=1 --load_model=True  --load_path=$baseline --do_test=True --test_data=$data --batch_size=16

echo $preTrain
CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data=$data1 --batch_size=16


echo $nerTrain
CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$nerTrain --do_test=True --test_data=$data --batch_size=16




data=./cc_data/chinese_spell_lower_4.txt
data1=./cc_data/chinese_spell_uniform_4.txt
# xaioxue
echo $baseline
CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$baseline --do_test=True --test_data=$data --batch_size=16

echo $preTrain
CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data=$data1  --batch_size=16

echo $nerTrain
CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$nerTrain --do_test=True --test_data=$data  --batch_size=16


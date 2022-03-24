#!/usr/bin/env bash
# 使用论文公开的模型进行测试，分析



#baseline="/data_local/TwoWaysToImproveCSC/BERT/save/baseline/sighan13/model.pkl"
#preTrain="/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/sighan13_no_ner/model.pkl"
#nerTrain="/data_local/TwoWaysToImproveCSC/BERT/save/nertrain/sighan13/model.pkl"

# /data_local/TwoWaysToImproveCSC/BERT/save/test/wiki_00_base_asame
# /data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_mask/sighan13/
baseline="/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_mask/epoch1.pkl"
preTrain="/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_mask/sighan13/model.pkl"
nerTrain="/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_mask/epoch2.pkl"
nerTrain2="/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_mask/sighan13_2/model.pkl"



gpu=0



data=./data/13test_lower.txt
task=test

#
#CUDA_VISIBLE_DEVICES=$gpu python new_bft_train1.py  --task_name=$task --gpu_num=1 --load_model=True  --load_path="/data_local/TwoWaysToImproveCSC/BERT/save/test/wiki_00_base_detect/epoch1.pkl" --do_test=True --test_data=$data --batch_size=16
#

# sighan13
echo $baseline
CUDA_VISIBLE_DEVICES=$gpu python bft_eval.py  --task_name=$task --gpu_num=1 --load_model=True  --load_path=$baseline --do_test=True --test_data=$data --batch_size=16

echo $preTrain
CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data=$data --batch_size=16


echo $nerTrain
CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$nerTrain --do_test=True --test_data=$data --batch_size=16

echo $nerTrain2
CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$nerTrain2 --do_test=True --test_data=$data --batch_size=16




data=./cc_data/chinese_spell_lower_4.txt
# xaioxue
echo $baseline
CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$baseline --do_test=True --test_data=$data --batch_size=16

echo $preTrain
CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data=$data  --batch_size=16

echo $nerTrain
CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$nerTrain --do_test=True --test_data=$data  --batch_size=16


echo $nerTrain2
CUDA_VISIBLE_DEVICES=$gpu  python bft_eval.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$nerTrain2 --do_test=True --test_data=$data  --batch_size=16


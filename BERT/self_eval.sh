#!/usr/bin/env bash
# 使用论文公开的模型进行测试，分析



#baseline="/data_local/TwoWaysToImproveCSC/BERT/save/baseline/sighan13/model.pkl"
#preTrain="/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/sighan13_no_ner/model.pkl"
#nerTrain="/data_local/TwoWaysToImproveCSC/BERT/save/nertrain/sighan13/model.pkl"

# /data_local/TwoWaysToImproveCSC/BERT/save/test/wiki_00_base_asame
# /data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_mask/sighan13/
# baseline0="/data_local/TwoWaysToImproveCSC/BERT/save/3090-pretrain/sighan13/model.pkl"
baseline="/data_local/TwoWaysToImproveCSC/BERT/save/test/test_macbert/base_no_ig/model.pkl"
preTrain="/data_local/TwoWaysToImproveCSC/BERT/save/test/test_macbert/mac_no_ig/model.pkl"
nerTrain="/data_local/TwoWaysToImproveCSC/BERT/save/test/test_macbert/base_mlm_task/model.pkl"
nerTrain0="/data_local/TwoWaysToImproveCSC/BERT/save/test/test_macbert/mac_mlm_task/model.pkl"

# nerTrain0="/data_local/TwoWaysToImproveCSC/BERT/save/test/test_macbert/mac_mlm_task/model.pkl"


gpu=7


data=./data/13test_lower.txt
task=test
# new_pretrain_auto.dev
# sighan13
echo $baseline
CUDA_VISIBLE_DEVICES=$gpu python new_bft_train_base.py  --task_name=$task --gpu_num=1 --load_model=True  --load_path=$baseline --do_test=True --test_data=$data --batch_size=16

echo $preTrain
CUDA_VISIBLE_DEVICES=$gpu  python new_bft_train_base_mac.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data=$data --batch_size=16


echo $nerTrain
CUDA_VISIBLE_DEVICES=$gpu  python new_bft_train_base_mlm.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$nerTrain --do_test=True --test_data=$data --batch_size=16


echo $nerTrain0
CUDA_VISIBLE_DEVICES=$gpu  python new_bft_train_base_mac_mlm.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$nerTrain0 --do_test=True --test_data=$data --batch_size=16



data=./cc_data/chinese_spell_lower_4.txt

## xaioxue
echo $baseline
CUDA_VISIBLE_DEVICES=$gpu  python new_bft_train_base.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$baseline --do_test=True --test_data=$data --batch_size=16

echo $preTrain
CUDA_VISIBLE_DEVICES=$gpu  python new_bft_train_base_mac.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data=$data  --batch_size=16

echo $nerTrain
CUDA_VISIBLE_DEVICES=$gpu  python new_bft_train_base_mlm.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$nerTrain --do_test=True --test_data=$data  --batch_size=16


echo $nerTrain0
CUDA_VISIBLE_DEVICES=$gpu  python new_bft_train_base_mac_mlm.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$nerTrain0 --do_test=True --test_data=$data  --batch_size=16


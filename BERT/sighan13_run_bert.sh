#!/usr/bin/env bash

# 仅仅使用自己的训练集
#CUDA_VISIBLE_DEVICES="0,1" python bft_train.py --task_name=sighan15 --gpu_num=2 --load_model=False --do_train=True --train_data=./data/15train.txt --do_valid=True --valid_data=./data/15test.txt --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=./save/sighan15_only/ --seed=20


# 在wang18训练的模型基础上进行训练
# path="/data_local/TwoWaysToImproveCSC/BERT/save/baseline/initial/epoch1.pkl"
# 在sighan2013上表现最好
# path="/data_local/TwoWaysToImproveCSC/BERT/save/baseline/initial/model.pkl"
# 在wang2018的验证集上表现最好
# baseline="/data_local/TwoWaysToImproveCSC/BERT/save/baseline/sighan13/"

# pretrain
#idx=13
#path=/data_local/TwoWaysToImproveCSC/BERT/save/baseline/wang2018_punct/model.pkl
#pretrain=/data_local/TwoWaysToImproveCSC/BERT/save/baseline/wang2018_punct/sighan${idx}/
#CUDA_VISIBLE_DEVICES="4,5" python bft_train.py --task_name=sighan13 --gpu_num=2 --load_model=True --load_path=$path --do_train=True --train_data=./data/punct_data/${idx}train.txt --do_valid=True --valid_data=./data/punct_data/${idx}test.txt --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=$pretrain --seed=10 > $pretrain/sighan${idx}_run_bert_epoch1.log 2>&1 &

#idx=13
#path=/data_local/TwoWaysToImproveCSC/BERT/save/test/wiki_00_base_detect/epoch1.pkl
#pretrain=/data_local/TwoWaysToImproveCSC/BERT/save/test/wiki_00_base_detect/sighan${idx}/
#CUDA_VISIBLE_DEVICES="4,5" python new_bft_train1.py  --task_name=sighan13 --gpu_num=2 --load_model=True --load_path=$path --do_train=True --train_data=./data/${idx}train_lower.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=$pretrain --seed=10 > $pretrain/sighan${idx}_run_bert_epoch1.log 2>&1 &
#

idx=13
path=/data_local/TwoWaysToImproveCSC/BERT/save/test/wiki_00_base_detect_joint/epoch2.pkl
pretrain=/data_local/TwoWaysToImproveCSC/BERT/save/test/wiki_00_base_detect_joint//sighan${idx}_2/
CUDA_VISIBLE_DEVICES="0,1" python bft_train_detect.py --task_name=test --gpu_num=2 --load_model=True --load_path=$path --do_train=True --train_data=./data/${idx}train_lower.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=$pretrain --seed=10 > $pretrain/sighan${idx}_run_bert_epoch1.log 2>&1 &

idx=13
path=/data_local/TwoWaysToImproveCSC/BERT/save/test/wiki_00_base_detect/epoch2.pkl
pretrain=/data_local/TwoWaysToImproveCSC/BERT/save/test/wiki_00_base_detect//sighan${idx}_2/
CUDA_VISIBLE_DEVICES="5,6" python bft_train_detect_joint.py --task_name=test --gpu_num=2 --load_model=True --load_path=$path --do_train=True --train_data=./data/${idx}train_lower.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=$pretrain --seed=10 > $pretrain/sighan${idx}_run_bert_epoch1.log 2>&1 &


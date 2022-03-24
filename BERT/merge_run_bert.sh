#!/usr/bin/env bash

# 使用merge数据训练,指定初始模型

# 不使用初始模型

idx=13
pretrain=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_acc_gradient/new_1/
CUDA_VISIBLE_DEVICES="4,5" python bft_train.py --task_name=sighan13 --gpu_num=2  --load_model=False --do_train=True --train_data=./data/merge_train.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=$pretrain --seed=111 > $pretrain/sighan${idx}_run_bert_epoch1.log 2>&1 &


idx=13
pretrain=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_acc_gradient/new_2/
CUDA_VISIBLE_DEVICES="6" python bft_train_gc.py --task_name=sighan13 --gpu_num=1  --gradient_accumulation_steps=2 --load_model=False --do_train=True --train_data=./data/merge_train.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=20 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$pretrain --seed=111 > $pretrain/sighan${idx}_run_bert_epoch1.log 2>&1 &


idx=13
pretrain=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_acc_gradient/new_5/
CUDA_VISIBLE_DEVICES="7" python bft_train_gc.py --task_name=sighan13 --gpu_num=1  --gradient_accumulation_steps=5 --load_model=False --do_train=True --train_data=./data/merge_train.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=20 --batch_size=10 --learning_rate=5e-5 --do_save=True --save_dir=$pretrain --seed=111 > $pretrain/sighan${idx}_run_bert_epoch1.log 2>&1 &



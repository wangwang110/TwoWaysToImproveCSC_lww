#!/usr/bin/env bash
# 
#CUDA_VISIBLE_DEVICES="0,1" python bft_train.py --task_name=wang18 --gpu_num=2 --load_model=False --do_train=True --train_data=./data/autog_wang_train.txt --do_valid=True --valid_data=./data/autog_wang_1k_dev.txt --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=./save/wang18/ --seed=20

# 在预训练模型的基础上训练
CUDA_VISIBLE_DEVICES="0,1" python bft_train.py --task_name=wang18 --gpu_num=2 --load_model=True --load_path="./save/pretrain/epoch1.pkl" --do_train=True --train_data=./data/rep_autog_wang_train.txt --do_valid=True --valid_data=./data/rep_autog_wang_1k_dev.txt --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=./save/wang18/ --seed=20
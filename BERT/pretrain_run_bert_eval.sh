#!/usr/bin/env bash

echo "pretrain_auto.dev"

CUDA_VISIBLE_DEVICES="1" python bft_train.py --task_name=pretrain --gpu_num=1 --load_model=True  --load_path="./save/pretrain/epoch1.pkl" --do_test=True --test_data=./data/pretrain_auto.dev --batch_size=8



echo "autog_wang_1k_dev"

CUDA_VISIBLE_DEVICES="1" python bft_train.py --task_name=pretrain --gpu_num=1 --load_model=True  --load_path="./save/pretrain/epoch1.pkl" --do_test=True --test_data=./data/autog_wang_1k_dev.txt --batch_size=8



echo "15test.txt"

CUDA_VISIBLE_DEVICES="1" python bft_train.py --task_name=pretrain --gpu_num=1 --load_model=True  --load_path="./save/pretrain/epoch1.pkl" --do_test=True --test_data=./data/15test.txt --batch_size=8



# autog_wang_1k_dev.txt
# 15train.txt
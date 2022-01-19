#!/usr/bin/env bash

echo "autog_wang_1k_dev"

CUDA_VISIBLE_DEVICES="1" python bft_train.py --task_name=wang18 --gpu_num=1 --load_model=True  --load_path="./save/wang18/model_modify.pkl" --do_test=True --test_data=./data/autog_wang_1k_dev.txt --batch_size=8


CUDA_VISIBLE_DEVICES="1" python bft_train.py --task_name=wang18 --gpu_num=1 --load_model=True  --load_path="./save/wang18/model.pkl" --do_test=True --test_data=./data/autog_wang_1k_dev.txt --batch_size=8

#for((i=1;i<10;i++));
#do
#CUDA_VISIBLE_DEVICES="1" python bft_train.py --task_name=wang18 --gpu_num=1 --load_model=True   --load_path=./save/wang18/epoch$i.pkl --do_test=True --test_data=./data/autog_wang_1k_dev.txt --batch_size=8
#done


echo "15test.txt"

CUDA_VISIBLE_DEVICES="1" python bft_train.py --task_name=wang18 --gpu_num=1 --load_model=True  --load_path="./save/wang18/model_modify.pkl" --do_test=True --test_data=./data/15test.txt --batch_size=8


CUDA_VISIBLE_DEVICES="1" python bft_train.py --task_name=wang18 --gpu_num=1 --load_model=True  --load_path="./save/wang18/model.pkl" --do_test=True --test_data=./data/15test.txt --batch_size=8

#for((i=1;i<10;i++));
#do
#CUDA_VISIBLE_DEVICES="1" python bft_train.py --task_name=wang18 --gpu_num=1 --load_model=True   --load_path=./save/wang18/epoch$i.pkl --do_test=True --test_data=./data/15test.txt --batch_size=8
#done


# autog_wang_1k_dev.txt
# 15train.txt
#!/usr/bin/env bash

# 仅仅使用自己的训练集
#CUDA_VISIBLE_DEVICES="0,1" python bft_train.py --task_name=sighan15 --gpu_num=2 --load_model=False --do_train=True --train_data=./data/15train.txt --do_valid=True --valid_data=./data/15test.txt --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=./save/sighan15_only/ --seed=20

# 在wang18训练的模型基础上进行训练
CUDA_VISIBLE_DEVICES="4" python bft_train.py --task_name=sighan15 --gpu_num=2 --load_model=True --load_path="./save/wang18/model.pkl" --do_train=True --train_data=./data/15train.txt --do_valid=True --valid_data=./data/15test.txt --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=./save/baseline/sighan15/ --seed=20
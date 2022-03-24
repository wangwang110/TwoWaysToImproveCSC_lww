#!/usr/bin/env bash

#
#name=pretrain_task
#save=/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_loss/
#CUDA_VISIBLE_DEVICES="0,1" python new_bft_train.py --task_name=$name --gpu_num=2 --load_model=False --do_train=True --train_data=./data/new_pretrain_all.train --do_valid=True --valid_data=./data/new_pretrain_all.dev --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=$save --seed=20 > $save/bft_train.log 2>&1 &
#



name=pretrain_task
save=/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_gc_2/
# CUDA_VISIBLE_DEVICES="0,1" python bft_train.py --task_name=$name --gpu_num=2 --load_model=False --do_train=True --train_data=./data/new_pretrain_all.train --do_valid=True --valid_data=./data/new_pretrain_all.dev --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=$save --seed=20 > $save/bft_train.log 2>&1 &


CUDA_VISIBLE_DEVICES="7" python bft_train_gc.py --task_name=$name --gpu_num=1 --gradient_accumulation_steps=2 --load_model=False --do_train=True --train_data=./data/new_pretrain_all.train --do_valid=True --valid_data=./data/new_pretrain_all.dev --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$save --seed=10 > $save/bft_train.log 2>&1 &

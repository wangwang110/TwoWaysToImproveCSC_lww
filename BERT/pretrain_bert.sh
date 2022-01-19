#!/usr/bin/env bash
name=pretrain
CUDA_VISIBLE_DEVICES="0,1,2" python bft_train.py --task_name=$name --gpu_num=3 --load_model=False --do_train=True --train_data=./data/ner_pretrain_auto.train --do_valid=True --valid_data=./data/ner_pretrain_auto.dev --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=./save/pretrain/initial/ --seed=20
#!/usr/bin/env bash


base=/data_local/TwoWaysToImproveCSC/BERT/save/test/
bert_path=/data_local/plm_models/chinese_L-12_H-768_A-12/
valid_data=./data/13test.txt
train_data=./data/wiki_00_base_1.train
save_path=$base/wiki_00_base_1/
mkdir $save_path
CUDA_VISIBLE_DEVICES="5" python bft_pretrain_mlm.py  --task_name=test --gpu_num=1 --load_model=False --do_train=True --train_data=$train_data --bert_path=$bert_path --do_valid=True --valid_data=$valid_data --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=100 > $save_path/bft_train.log 2>&1 &


train_data=./data/wiki_00_base_2.train
save_path=$base/wiki_00_base_2/
mkdir $save_path
CUDA_VISIBLE_DEVICES="6" python bft_pretrain_mlm.py  --task_name=test --gpu_num=1 --load_model=False --do_train=True --train_data=$train_data --bert_path=$bert_path --do_valid=True --valid_data=$valid_data --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=100 > $save_path/bft_train.log 2>&1 &


train_data=./data/wiki_00_word_0.train
save_path=$base/wiki_00_word_0/
mkdir $save_path
CUDA_VISIBLE_DEVICES="7" python bft_pretrain_mlm.py  --task_name=test --gpu_num=1 --load_model=False --do_train=True --train_data=$train_data --bert_path=$bert_path --do_valid=True --valid_data=$valid_data --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=100 > $save_path/bft_train.log 2>&1 &

#!/usr/bin/env bash


path=./save/test/wiki_00_base_2/epoch1.pkl
base=./save/test/wiki_00_base_2/
bert_path=/data_local/plm_models/chinese_L-12_H-768_A-12/
batchsize=20
idx=13
gpu=5

name=sighan13
train_data=./data/13train.txt
save_path=${base}/${name}/
mkdir $save_path
echo $save_path
CUDA_VISIBLE_DEVICES=$gpu  python bft_pretrain_mlm.py  --bert_path=$bert_path  --ignore_sep=False --task_name=test --gpu_num=1 --load_model=True --load_path=$path --do_train=True --train_data=$train_data  --do_valid=True --valid_data=./data/${idx}test.txt --epoch=10 --batch_size=$batchsize --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=10

name=sighan13cc
train_data=./data/13train_cc_shuf.txt
save_path=${base}/${name}/
mkdir $save_path
echo $save_path
CUDA_VISIBLE_DEVICES=$gpu  python bft_pretrain_mlm.py   --bert_path=$bert_path --ignore_sep=False --task_name=test --gpu_num=1 --load_model=True --load_path=$path --do_train=True --train_data=$train_data  --do_valid=True --valid_data=./data/${idx}test.txt --epoch=10 --batch_size=$batchsize  --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=10


name=sighan13_700
train_data=./data/13train_700.txt
save_path=${base}/${name}/
mkdir $save_path
echo $save_path
CUDA_VISIBLE_DEVICES=$gpu  python bft_pretrain_mlm.py   --bert_path=$bert_path --ignore_sep=False --task_name=test --gpu_num=1 --load_model=True --load_path=$path --do_train=True --train_data=$train_data  --do_valid=True --valid_data=./data/${idx}test.txt --epoch=10 --batch_size=$batchsize  --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=10



name=merge
train_data=./data/merge_train.txt
save_path=${base}/${name}/
mkdir $save_path
echo $save_path
CUDA_VISIBLE_DEVICES=$gpu  python bft_pretrain_mlm.py   --bert_path=$bert_path --ignore_sep=False --task_name=test --gpu_num=1 --load_model=True --load_path=$path --do_train=True --train_data=$train_data  --do_valid=True --valid_data=./data/${idx}test.txt --epoch=10 --batch_size=$batchsize  --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=10

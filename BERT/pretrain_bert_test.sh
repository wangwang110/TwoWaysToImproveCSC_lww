#!/usr/bin/env bash



train_data=./data/rep_autog_wang_train.txt
valid_data=./data/rep_autog_wang_1k_dev.txt
#path=/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_mask/epoch1.pkl
#
#save_path=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_acc_gradient/new_margin/
#CUDA_VISIBLE_DEVICES="0,1" python bft_train_margin.py --task_name=test --gpu_num=2 --load_model=True --load_path=$path --do_train=True --train_data=$train_data --do_valid=True --valid_data=$valid_data --epoch=10 --batch_size=20 --learning_rate=2e-5  --do_save=True --save_dir=$save_path --seed=10 > $save_path/bft_train.log 2>&1 &

#
#save_path=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_acc_gradient/new_2/
#CUDA_VISIBLE_DEVICES="4" python bft_train_gc.py --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 --load_model=True --load_path=$path --do_train=True --train_data=$train_data --do_valid=True --valid_data=$valid_data --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=10 > $save_path/bft_train.log 2>&1 &
#


#save_path=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_acc_gradient/new_6/
#CUDA_VISIBLE_DEVICES="4" python bft_train_gc.py --task_name=test --gpu_num=1 --gradient_accumulation_steps=6 --load_model=True --load_path=$path --do_train=True --train_data=$train_data --do_valid=True --valid_data=$valid_data --epoch=10 --batch_size=10 --learning_rate=5e-5 --do_save=True --save_dir=$save_path --seed=10 > $save_path/bft_train.log 2>&1 &
#

#save_path=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_multi_task/wang2018/
#CUDA_VISIBLE_DEVICES="0" python new_bft_train.py  --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 --load_model=False --do_train=True --train_data=$train_data --do_valid=True --valid_data=$valid_data --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=10 > $save_path/bft_train.log 2>&1 &
##

#save_path=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_multi_task/wang2018_tokenize/
#CUDA_VISIBLE_DEVICES="0" python new_bft_train0.py  --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 --load_model=False --do_train=True --train_data=$train_data --do_valid=True --valid_data=$valid_data --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=10 > $save_path/bft_train.log 2>&1 &
##

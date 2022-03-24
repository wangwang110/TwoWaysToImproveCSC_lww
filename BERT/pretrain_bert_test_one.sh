#!/usr/bin/env bash



#train_data=./data/wiki_00_base_trans_only.train
#save_path=/data_local/TwoWaysToImproveCSC/BERT/save/test/wiki_00_base_trans_only/
#CUDA_VISIBLE_DEVICES="2,3" python bft_train.py --task_name=sighan13 --gpu_num=2 --load_model=False --do_train=True --train_data=$train_data --do_valid=True --valid_data=./data/13test_lower.txt --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=10 > $save_path/bft_train.log 2>&1 &
#



#train_data=./data/wiki_00_base.train
#save_path=/data_local/TwoWaysToImproveCSC/BERT/save/test/wiki_00_base_asame/
#CUDA_VISIBLE_DEVICES="0,1" python bft_train.py --task_name=pretrain_test --gpu_num=2 --load_model=False --do_train=True --train_data=$train_data --do_valid=True --valid_data=./data/13test_lower.txt --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=10 > $save_path/bft_train.log 2>&1 &
#

train_data=./data/wiki_00_base.train
save_path=/data_local/TwoWaysToImproveCSC/BERT/save/test/wiki_00_base_one_gpu/
CUDA_VISIBLE_DEVICES="2" python new_bft_train.py --task_name=pretrain_test --gpu_num=1 --load_model=False --do_train=True --train_data=$train_data --do_valid=True --valid_data=./data/13test_lower.txt --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=10 > $save_path/bft_train.log 2>&1 &




#train_data=./data/wiki_00_base_confuse
#save_path=/data_local/TwoWaysToImproveCSC/BERT/save/test/wiki_00_base_confuse/
#CUDA_VISIBLE_DEVICES="6,7" python bft_train.py --task_name=sighan13 --gpu_num=2 --load_model=False --do_train=True --train_data=$train_data --do_valid=True --valid_data=./data/13test_lower.txt --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=10 > $save_path/bft_train.log 2>&1 &
#



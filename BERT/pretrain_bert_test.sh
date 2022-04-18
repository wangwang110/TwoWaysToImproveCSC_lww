#!/usr/bin/env bash



train_data=./data/merge_train.txt
valid_data=./data/13test_lower.txt


save_path=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_macbert/base_no_ig/
CUDA_VISIBLE_DEVICES="0" python new_bft_train_base.py  --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 --load_model=False --do_train=True --train_data=$train_data --do_valid=True --valid_data=$valid_data --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=100 > $save_path/bft_train.log 2>&1 &

save_path=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_macbert/mac_no_ig/
CUDA_VISIBLE_DEVICES="1" python new_bft_train_base_mac.py  --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 --load_model=False --do_train=True --train_data=$train_data --do_valid=True --valid_data=$valid_data --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=100 > $save_path/bft_train.log 2>&1 &


save_path=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_macbert/mac_mlm_task/
CUDA_VISIBLE_DEVICES="2" python new_bft_train_base_mac_mlm_task.py  --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 --load_model=False --do_train=True --train_data=$train_data --do_valid=True --valid_data=$valid_data --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=100 > $save_path/bft_train.log 2>&1 &


save_path=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_macbert/base_mlm_task/
CUDA_VISIBLE_DEVICES="3" python new_bft_train_base_mlm_task.py  --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 --load_model=False --do_train=True --train_data=$train_data --do_valid=True --valid_data=$valid_data --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=100 > $save_path/bft_train.log 2>&1 &


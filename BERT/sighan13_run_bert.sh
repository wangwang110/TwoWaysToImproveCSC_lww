#!/usr/bin/env bash

# 仅仅使用自己的训练集
#CUDA_VISIBLE_DEVICES="0,1" python bft_train.py --task_name=sighan15 --gpu_num=2 --load_model=False --do_train=True --train_data=./data/15train.txt --do_valid=True --valid_data=./data/15test.txt --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=./save/sighan15_only/ --seed=20


# 在wang18训练的模型基础上进行训练
# path="/data_local/TwoWaysToImproveCSC/BERT/save/baseline/initial/epoch1.pkl"
# 在sighan2013上表现最好
# path="/data_local/TwoWaysToImproveCSC/BERT/save/baseline/initial/model.pkl"
# 在wang2018的验证集上表现最好
# baseline="/data_local/TwoWaysToImproveCSC/BERT/save/baseline/sighan13/"

# pretrain
#idx=13
#path=/data_local/TwoWaysToImproveCSC/BERT/save/baseline/wang2018_punct/model.pkl
#pretrain=/data_local/TwoWaysToImproveCSC/BERT/save/baseline/wang2018_punct/sighan${idx}/
#CUDA_VISIBLE_DEVICES="4,5" python bft_train.py --task_name=sighan13 --gpu_num=2 --load_model=True --load_path=$path --do_train=True --train_data=./data/punct_data/${idx}train.txt --do_valid=True --valid_data=./data/punct_data/${idx}test.txt --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=$pretrain --seed=10 > $pretrain/sighan${idx}_run_bert_epoch1.log 2>&1 &

#idx=13
#path=/data_local/TwoWaysToImproveCSC/BERT/save/epoch1_step6200.pkl
#pretrain=/data_local/TwoWaysToImproveCSC/BERT/save/3090-pretrain/sighan${idx}/
#CUDA_VISIBLE_DEVICES="0,1" python bft_adtrain.py  --task_name=sighan13 --gpu_num=2 --load_model=True --load_path=$path --do_train=True --train_data=./data/${idx}train_lower.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=$pretrain --seed=10 > $pretrain/sighan${idx}_run_bert_epoch1.log 2>&1 &
#

#idx=13
#path=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_multi_task/wang2018_tokenize/model.pkl
#pretrain=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_multi_task/wang2018_tokenize/sighan${idx}/
#CUDA_VISIBLE_DEVICES="0" python new_bft_train_base.py  --task_name=sighan13 --gpu_num=1 --gradient_accumulation_steps=2  --load_model=False --do_train=True --train_data=./data/merge_train.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$pretrain --seed=10 > $pretrain/sighan${idx}_run_bert_epoch1.log 2>&1 &
#
#idx=13
#path=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_multi_task/wang2018/model.pkl
#pretrain=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_multi_task/wang2018/sighan${idx}/
#CUDA_VISIBLE_DEVICES="1" python new_bft_train.py  --task_name=sighan13 --gpu_num=1 --gradient_accumulation_steps=2  --load_model=False --do_train=True --train_data=./data/merge_train.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$pretrain --seed=10 > $pretrain/sighan${idx}_run_bert_epoch1.log 2>&1 &
#
#
#idx=13
#path=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_multi_task/wang2018_loss_ignore/model.pkl
#pretrain=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_multi_task/wang2018_loss_ignore/sighan${idx}/
#CUDA_VISIBLE_DEVICES="2" python new_bft_train0.py  --task_name=sighan13 --gpu_num=1 --gradient_accumulation_steps=2  --load_model=False --do_train=True --train_data=./data/merge_train.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$pretrain --seed=10 > $pretrain/sighan${idx}_run_bert_epoch1.log 2>&1 &
#
#
#
#idx=13
#path=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_multi_task/wang2018_seq/model.pkl
#pretrain=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_multi_task/wang2018_seq/sighan${idx}/
#CUDA_VISIBLE_DEVICES="6" python new_bft_train1.py  --task_name=sighan13 --gpu_num=1 --gradient_accumulation_steps=2  --load_model=False --do_train=True --train_data=./data/merge_train.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$pretrain --seed=10 > $pretrain/sighan${idx}_run_bert_epoch1.log 2>&1 &
#
#
#
#idx=13
#path=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_multi_task/wang2018_seq_pos/model.pkl
#pretrain=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_multi_task/wang2018_seq_pos/sighan${idx}/
#CUDA_VISIBLE_DEVICES="7" python new_bft_train2.py  --task_name=sighan13 --gpu_num=1 --gradient_accumulation_steps=2  --load_model=False --do_train=True --train_data=./data/merge_train.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$pretrain --seed=10 > $pretrain/sighan${idx}_run_bert_epoch1.log 2>&1 &
#


#idx=13
#path=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_multi_task/wang2018_seq_pos/model.pkl
#pretrain=/data_local/TwoWaysToImproveCSC/BERT/save/test/test_multi_task/wang2018_seq_pos/sighan${idx}/
#CUDA_VISIBLE_DEVICES="7" python new_bft_train2.py  --task_name=sighan13 --gpu_num=1 --gradient_accumulation_steps=2  --load_model=False --do_train=True --train_data=./data/merge_train.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$pretrain --seed=10 > $pretrain/sighan${idx}_run_bert_epoch1.log 2>&1 &
#
#
#idx=13
#path=/data_local/TwoWaysToImproveCSC/BERT/save/epoch1_step6200.pkl
#pretrain=/data_local/TwoWaysToImproveCSC/BERT/save/3090-pretrain/sighan${idx}/
#CUDA_VISIBLE_DEVICES="0,1" python bft_adtrain.py  --task_name=sighan13 --gpu_num=2 --load_model=True --load_path=$path --do_train=True --train_data=./data/wiki_00_base_trg.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=10 --batch_size=20 --learning_rate=2e-5 --do_save=True --save_dir=$pretrain --seed=10 > $pretrain/sighan${idx}_run_bert_epoch1.log 2>&1 &
#
#
#idx=13
#path=/data_local/TwoWaysToImproveCSC/BERT/save/epoch1_step6200.pkl
#
#train_data=/data_local/chinese_data/pretain_remove_short.data
#save_path=/data_local/TwoWaysToImproveCSC/BERT/save/3090-pretrain/all/
#CUDA_VISIBLE_DEVICES="0,1" python bft_pretrain.py --task_name=pretrain_test --gpu_num=2 --load_model=False --do_train=True --train_data=$train_data --do_valid=True --valid_data=./data/13test_lower.txt --epoch=10 --batch_size=48 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=10 > $save_path/bft_train.log 2>&1 &
#
#

#idx=13
#path=/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_gc_2/epoch1.pkl
#save_path=/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_gc_2/add_ctc/
#CUDA_VISIBLE_DEVICES="0" python bft_train_gc.py  --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 --load_model=True --load_path=$path --do_train=True --train_data=./data/ctc/qua_csc.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=10 > $save_path/bft_train.log 2>&1 &
##
#
#
#idx=13
#path=/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_gc_2/epoch1.pkl
#save_path=/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_gc_2/sighan${idx}/
#CUDA_VISIBLE_DEVICES="1" python bft_train_gc.py  --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 --load_model=True --load_path=$path --do_train=True --train_data=./data/${idx}train_lower.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=10 > $save_path/bft_train.log 2>&1 &
##
#
#
#idx=13
#path=/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_gc_2/epoch2.pkl
#save_path=/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_gc_2/sighan${idx}_2/
#CUDA_VISIBLE_DEVICES="2" python bft_train_gc.py  --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 --load_model=True --load_path=$path --do_train=True --train_data=./data/${idx}train_lower.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=10 > $save_path/bft_train.log 2>&1 &
##


idx=13
path=/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_gc_2/add_ctc/model.pkl
save_path=/data_local/TwoWaysToImproveCSC/BERT/save/pretrain/base_998_gc_2/add_ctc/sighan13/
CUDA_VISIBLE_DEVICES="0" python bft_train_gc.py  --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 --load_model=True --load_path=$path --do_train=True --train_data=./data/${idx}train_lower.txt --do_valid=True --valid_data=./data/${idx}test_lower.txt --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=10 > $save_path/bft_train.log 2>&1 &
#

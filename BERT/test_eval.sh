#!/usr/bin/env bash



name=sighan13
base_path=./save/test/wiki_00_word_0/
preTrain=$base_path/epoch1.pkl
preTrain0=$base_path/sighan13/model.pkl
preTrain1=$base_path/sighan13_700/model.pkl
preTrain2=$base_path/sighan13cc/model.pkl
preTrain3=$base_path/merge/model.pkl

bert_path=/data_local/plm_models/chinese_L-12_H-768_A-12/



gpu=6


data=./data/13test.txt
task=test


echo $preTrain
CUDA_VISIBLE_DEVICES=$gpu  python bft_pretrain_mlm.py  --bert_path=$bert_path  --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data=$data --batch_size=16


echo $preTrain0
CUDA_VISIBLE_DEVICES=$gpu  python bft_pretrain_mlm.py  --bert_path=$bert_path --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain0 --do_test=True --test_data=$data --batch_size=16


echo $preTrain1
CUDA_VISIBLE_DEVICES=$gpu  python bft_pretrain_mlm.py  --bert_path=$bert_path  --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain1 --do_test=True --test_data=$data --batch_size=16
#
echo $preTrain2
CUDA_VISIBLE_DEVICES=$gpu  python bft_pretrain_mlm.py  --bert_path=$bert_path --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain2 --do_test=True --test_data=$data --batch_size=16
#


echo $preTrain3
CUDA_VISIBLE_DEVICES=$gpu  python bft_pretrain_mlm.py   --bert_path=$bert_path --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain3 --do_test=True --test_data=$data --batch_size=16
#


echo "==============================================="

data=./data/chinese_spell_lower_4.txt

echo $preTrain
CUDA_VISIBLE_DEVICES=$gpu  python bft_pretrain_mlm.py   --bert_path=$bert_path --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data=$data  --batch_size=16


echo $preTrain0
CUDA_VISIBLE_DEVICES=$gpu  python bft_pretrain_mlm.py  --bert_path=$bert_path --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain0 --do_test=True --test_data=$data  --batch_size=16

echo $preTrain1
CUDA_VISIBLE_DEVICES=$gpu  python bft_pretrain_mlm.py  --bert_path=$bert_path --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain1 --do_test=True --test_data=$data  --batch_size=16
#

echo $preTrain2
CUDA_VISIBLE_DEVICES=$gpu  python bft_pretrain_mlm.py  --bert_path=$bert_path --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain2 --do_test=True --test_data=$data  --batch_size=16
#

echo $preTrain3
CUDA_VISIBLE_DEVICES=$gpu  python bft_pretrain_mlm.py   --bert_path=$bert_path --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain3 --do_test=True --test_data=$data  --batch_size=16
#

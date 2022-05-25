#!/usr/bin/env bash

# 使用merge数据训练, 测试不同变体
chinese_bert_path=/data_local/plm_models/chinese_L-12_H-768_A-12/
train_data=./data/merge_train.txt
valid_data1=./data/13test.txt
valid_data2=./cc_data/chinese_spell_lower_4.txt
epoch=10
batch_size=10
lr=2e-5
seed=200
gc_num=5


save_path=./save/merge_test/mlm/
mkdir $save_path
echo $save_path
echo "========training========"
CUDA_VISIBLE_DEVICES=$gc_num python bft_train_mlm.py --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 \
--load_model=False --bert_path=$chinese_bert_path --do_train=True --train_data=$train_data \
 --do_valid=True --valid_data=$valid_data1 --epoch=$epoch --batch_size=$batch_size --learning_rate=$lr \
 --do_save=True --save_dir=$save_path --seed=$seed --mlm --multitask_weight=0 \

echo "========testing========"

CUDA_VISIBLE_DEVICES=$gc_num python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data1 \
--mlm

CUDA_VISIBLE_DEVICES=$gc_num python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data2 \
--mlm


save_path=./save/merge_test/error_weight/
mkdir $save_path
echo $save_path
echo "========training========"
CUDA_VISIBLE_DEVICES=$gc_num python bft_train_mlm.py --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 \
--load_model=False --bert_path=$chinese_bert_path --do_train=True --train_data=$train_data \
 --do_valid=True --valid_data=$valid_data1 --epoch=$epoch --batch_size=$batch_size --learning_rate=$lr \
 --do_save=True --save_dir=$save_path --seed=$seed --mlm --multitask_weight=0 --error_weight=3 \

echo "========testing========"

CUDA_VISIBLE_DEVICES=$gc_num python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data1 \
--mlm

CUDA_VISIBLE_DEVICES=$gc_num python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data2 \
--mlm


save_path=./save/merge_test/mlm_cpo/
mkdir $save_path
echo $save_path
echo "========training========"
CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 \
--load_model=False --bert_path=$chinese_bert_path --do_train=True --train_data=$train_data \
 --do_valid=True --valid_data=$valid_data1 --epoch=$epoch --batch_size=$batch_size --learning_rate=$lr \
 --do_save=True --save_dir=$save_path --seed=$seed --mlm --multitask_weight=0 --cpoloss \
 # > $save_path/csc_train_mlm_tok.log 2>&1 &

echo "========testing========"

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data1 \
--mlm

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data2 \
--mlm

save_path=./save/merge_test/mlm_vocab/
mkdir $save_path
echo $save_path
echo "========training========"
CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 \
--load_model=False --bert_path=$chinese_bert_path --do_train=True --train_data=$train_data \
 --do_valid=True --valid_data=$valid_data1 --epoch=$epoch --batch_size=$batch_size --learning_rate=$lr \
 --do_save=True --save_dir=$save_path --seed=$seed --mlm --multitask_weight=0 --vocab_refine \
 # > $save_path/csc_train_mlm_tok.log 2>&1 &

echo "========testing========"

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data1 --vocab_refine \
--mlm

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data2 --vocab_refine \
--mlm


save_path=./save/merge_test/mlm_multi/
mkdir $save_path
echo $save_path
echo "========training========"
CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 \
--load_model=False --bert_path=$chinese_bert_path --do_train=True --train_data=$train_data \
 --do_valid=True --valid_data=$valid_data1 --epoch=$epoch --batch_size=$batch_size --learning_rate=$lr \
 --do_save=True --save_dir=$save_path --seed=$seed --mlm --multitask_weight=0.7 \
 # > $save_path/csc_train_mlm_tok.log 2>&1 &

echo "========testing========"

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data1 \
--mlm

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data2 \
--mlm


save_path=./save/merge_test/mlm_warmup/
mkdir $save_path
echo $save_path
echo "========training========"
CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 \
--load_model=False --bert_path=$chinese_bert_path --do_train=True --train_data=$train_data \
 --do_valid=True --valid_data=$valid_data1 --epoch=$epoch --batch_size=$batch_size --learning_rate=$lr \
 --do_save=True --save_dir=$save_path --seed=$seed --mlm --multitask_weight=0 --do_warmup \
 # > $save_path/csc_train_mlm_tok.log 2>&1 &

echo "========testing========"

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data1 \
--mlm

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data2 \
--mlm



save_path=./save/merge_test/mlm_warmup_cpo/
mkdir $save_path
echo $save_path
echo "========training========"
CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 \
--load_model=False --bert_path=$chinese_bert_path --do_train=True --train_data=$train_data \
 --do_valid=True --valid_data=$valid_data1 --epoch=$epoch --batch_size=$batch_size --learning_rate=$lr \
 --do_save=True --save_dir=$save_path --seed=$seed --mlm --multitask_weight=0 --do_warmup --cpoloss
 # > $save_path/csc_train_mlm_tok.log 2>&1 &

echo "========testing========"

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data1 \
--mlm

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data2 \
--mlm


############base############base############base############base############base############base############base############base############base############base############base############base

save_path=./save/merge_test/base/
mkdir $save_path
echo $save_path
echo "========training========"
CUDA_VISIBLE_DEVICES=$gc_num python bft_train_mlm.py --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 \
--load_model=False --bert_path=$chinese_bert_path --do_train=True --train_data=$train_data \
 --do_valid=True --valid_data=$valid_data1 --epoch=$epoch --batch_size=$batch_size --learning_rate=$lr \
 --do_save=True --save_dir=$save_path --seed=$seed --multitask_weight=0 \

echo "========testing========"

CUDA_VISIBLE_DEVICES=$gc_num python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data1

CUDA_VISIBLE_DEVICES=$gc_num python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data2


save_path=./save/merge_test/base_cpo/
mkdir $save_path
echo $save_path
echo "========training========"
CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 \
--load_model=False --bert_path=$chinese_bert_path --do_train=True --train_data=$train_data \
 --do_valid=True --valid_data=$valid_data1 --epoch=$epoch --batch_size=$batch_size --learning_rate=$lr \
 --do_save=True --save_dir=$save_path --seed=$seed  --multitask_weight=0 --cpoloss \
 # > $save_path/csc_train_mlm_tok.log 2>&1 &

echo "========testing========"

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data1

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data2


save_path=./save/merge_test/base_vocab/
mkdir $save_path
echo $save_path
echo "========training========"
CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 \
--load_model=False --bert_path=$chinese_bert_path --do_train=True --train_data=$train_data \
 --do_valid=True --valid_data=$valid_data1 --epoch=$epoch --batch_size=$batch_size --learning_rate=$lr \
 --do_save=True --save_dir=$save_path --seed=$seed  --multitask_weight=0 --vocab_refine \
 # > $save_path/csc_train_mlm_tok.log 2>&1 &

echo "========testing========"

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data1 --vocab_refine

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data2 --vocab_refine


save_path=./save/merge_test/base_multi/
mkdir $save_path
echo $save_path
echo "========training========"
CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 \
--load_model=False --bert_path=$chinese_bert_path --do_train=True --train_data=$train_data \
 --do_valid=True --valid_data=$valid_data1 --epoch=$epoch --batch_size=$batch_size --learning_rate=$lr \
 --do_save=True --save_dir=$save_path --seed=$seed --multitask_weight=0.7
 # > $save_path/csc_train_mlm_tok.log 2>&1 &

echo "========testing========"

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data1


CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data2

save_path=./save/merge_test/base_warmup/
mkdir $save_path
echo $save_path
echo "========training========"
CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 \
--load_model=False --bert_path=$chinese_bert_path --do_train=True --train_data=$train_data \
 --do_valid=True --valid_data=$valid_data1 --epoch=$epoch --batch_size=$batch_size --learning_rate=$lr \
 --do_save=True --save_dir=$save_path --seed=$seed  --multitask_weight=0 --do_warmup \
 # > $save_path/csc_train_mlm_tok.log 2>&1 &

echo "========testing========"

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data1

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data2

save_path=./save/merge_test/base_warmup_cpo/
mkdir $save_path
echo $save_path
echo "========training========"
CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 \
--load_model=False --bert_path=$chinese_bert_path --do_train=True --train_data=$train_data \
 --do_valid=True --valid_data=$valid_data1 --epoch=$epoch --batch_size=$batch_size --learning_rate=$lr \
 --do_save=True --save_dir=$save_path --seed=$seed  --multitask_weight=0 --do_warmup --cpoloss
 # > $save_path/csc_train_mlm_tok.log 2>&1 &

echo "========testing========"

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data1

CUDA_VISIBLE_DEVICES=$gc_num  python bft_train_mlm.py --task_name=test --gpu_num=1 \
--load_model=True --load_path=$save_path/model.pkl \
--bert_path=$chinese_bert_path --do_test=True --test_data=$valid_data2


#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES="1" python bft_eval.py --task_name=wang18 --gpu_num=1 --load_model=True  --load_path="./save/sighan13_best_model/model.pkl" --do_test=True --test_data="../large_data/weibo/weibo_clean_second.txt" --batch_size=16


CUDA_VISIBLE_DEVICES="4" python bft_eval.py --task_name=wang18 --gpu_num=1 --load_model=True  --load_path="./save/sighan13_best_model/model.pkl" --do_test=True --test_data="./cc_data/xiaoxue_tc_4_sent.txt" --batch_size=16


#!/usr/bin/env bash

data=/data_local/TwoWaysToImproveCSC/large_data/weibo
  
python generate_csc_data.py --input $data/weibo_correct_first.txt --output $data/weibo_correct_first_csc.txt --path  $data/weibo_vocab.pkl

python generate_csc_data.py --input $data/weibo_correct_second.txt --output $data/weibo_correct_second_csc.txt --path  $data/weibo_vocab.pkl
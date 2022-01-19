#!/usr/bin/env bash
data=/data_local/TwoWaysToImproveCSC/large_data

for((i=0;i<9;i++));  
do   
python generate_csc_data.py --input $data/zh_wiki_sent/wiki_0${i}_sent --output $data/zh_wiki_csc/wiki_0${i}_csc --path  $data/zh_wiki_sent/wiki_vocab.pkl
done
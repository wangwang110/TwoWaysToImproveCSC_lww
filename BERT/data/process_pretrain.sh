#!/usr/bin/env bash

data=/data_local/TwoWaysToImproveCSC/BERT/data/

wiki_data_path=/data_local/TwoWaysToImproveCSC/large_data/zh_wiki_csc_new/
weibo_data_path=/data_local/TwoWaysToImproveCSC/large_data/zh_weibo_new/


new2016zh=/data_local/TwoWaysToImproveCSC/large_data/new2016zh/news2016zh_cor_shuf.txt
translation2019zh=/data_local/TwoWaysToImproveCSC/large_data/translation2019zh/translation2019zh_cor_shuf.txt
webtext2019zh=/data_local/TwoWaysToImproveCSC/large_data/webtext2019zh/web_zh_cor_shuf.txt


#
## + lower + unk （原来的数据）
#
#cat $wiki_data_path/*  $weibo_data_path/* > $data/pretrain_all.txt
#
#shuf $data/pretrain_all.txt -o $data/pretrain_all_shuf.txt
#
#sed -n '1,10272000p' $data/pretrain_all_shuf.txt > $data/new_pretrain_auto.train
#sed -n '10272001,$p' $data/pretrain_all_shuf.txt > $data/new_pretrain_auto.dev
## 10272000,以前是10302401
#
#rm -rf $data/tmp*.txt



## + lower + unk （原来的数据 + 作文数据，比例怎么样呢？先来2份吧）
#cat $wiki_data_path/*  $weibo_data_path/*  $data/wiki_00_base_zuowen_2.train  > $data/pretrain_all.txt
#shuf $data/pretrain_all.txt -o $data/pretrain_all_shuf.txt
#sed -n '1,10272000p' $data/pretrain_all_shuf.txt > $data/new_pretrain_zuowen.train
#sed -n '10272001,$p' $data/pretrain_all_shuf.txt > $data/new_pretrain_zuowen.dev
## 10302401
#rm -rf $data/tmp*.txt



#
##  + lower + unk （所有数据）
#cat $wiki_data_path/*  $weibo_data_path/*  ./wiki_00_base_zuowen_2.train $new2016zh $translation2019zh $webtext2019zh  > $data/pretrain_all.txt
#shuf $data/pretrain_all.txt -o $data/pretrain_all_shuf.txt
#sed -n '1,10272000p' $data/pretrain_all_shuf.txt > $data/new_pretrain_all.train
#sed -n '10272001,$p' $data/pretrain_all_shuf.txt > $data/new_pretrain_all.dev
## 10302401
#rm -rf $data/tmp*.txt
#



#  + lower + unk （no wiki no weibo 数据）
cat ./wiki_00_base_zuowen_2.train $new2016zh $translation2019zh $webtext2019zh  > $data/pretrain_all.txt
shuf $data/pretrain_all.txt -o $data/pretrain_all_shuf.txt
sed -n '1,9186000p' $data/pretrain_all_shuf.txt > $data/new_pretrain_all.train
sed -n '9186001,$p' $data/pretrain_all_shuf.txt > $data/new_pretrain_all.dev
# 9188206
rm -rf $data/tmp*.txt






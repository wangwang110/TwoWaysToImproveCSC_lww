# -*- coding: utf-8 -*-
# get_similiar_char.py
import numpy as np
import cv2
import os
import pickle
from operator import itemgetter


def read_img_2_list(img_path):
    # 读取图片
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    # 把图片转换为灰度模式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(-1, 1)
    return [_[0] for _ in img.tolist()]


# 获取所有汉字的向量表示，以dict储存
def get_all_char_vectors():
    path = "/data_local/TwoWaysToImproveCSC/large_data/char_pic/"
    bin_path = os.path.dirname(path) + "/character.bin"
    if os.path.exists(bin_path):
        img_vector_dict = pickle.load(open(bin_path, "rb"))
    else:
        image_paths = [_ for _ in os.listdir(path) if _.endswith("png")]
        img_vector_dict = {}
        for image_path in image_paths:
            img_vector_dict[image_path[0]] = read_img_2_list(img_path=path + image_path)
        pickle.dump(img_vector_dict, open(bin_path, "wb"))
    return img_vector_dict


# 计算两个向量之间的余弦相似度
def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return dot_product / ((normA ** 0.5) * (normB ** 0.5))


if __name__ == '__main__':
    img_vector_dict = get_all_char_vectors()

    # 获取最接近的汉字
    similarity_dict = {}
    # while True:
    # match_char = input("输入汉字: ")
    match_char = "子"
    if match_char in img_vector_dict:
        match_vector = img_vector_dict[match_char]
        for char, vector in img_vector_dict.items():
            cosine_similar = cosine_similarity(match_vector, vector)
            similarity_dict[char] = cosine_similar
        # 按相似度排序，取前10个
        sorted_similarity = sorted(similarity_dict.items(), key=itemgetter(1), reverse=True)
        print([(char, round(similarity, 4)) for char, similarity in sorted_similarity[:10]])

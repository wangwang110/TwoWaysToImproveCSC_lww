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
    path = "/data_local/TwoWaysToImproveCSC/large_data/char_pic_more/"
    bin_path = os.path.dirname(path) + "/character_more.bin"
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

    pinyin_char = pickle.load(open("./data/same_pinyin_chars.pkl", "rb"))
    word_sets = set()
    for s in pinyin_char:
        for t in pinyin_char[s]:
            word_sets.add(t)
    print(len(word_sets))

    char_li = list(word_sets)

    val = 0.95
    with open("./tmp" + str(val) + ".txt", "w", encoding="utf=8") as f:
        for i, c1 in enumerate(char_li):
            # for c2 in char_li[i:]:
            #     if c1 != c2 and c1 in img_vector_dict and c2 in img_vector_dict:
            #         vector = img_vector_dict[c1]
            #         match_vector = img_vector_dict[c2]
            #         cosine_similar = cosine_similarity(vector, match_vector)
            #         if cosine_similar > val:
            #             f.write(c1 + "\t" + c2 + ":" + str(cosine_similar) + "\n")
            #             print(c1 + "\t" + c2 + ":" + str(cosine_similar) + "\n")
            #             # print(f'For character pair ({c1}, {c2}):')
            #             # print(f'    v-sim = {c.shape_similarity(c1, c2)}')
            #             # print(f'    p-sim = {c.pronunciation_similarity(c1, c2)}\n')

            if c1 in img_vector_dict:
                print(c1)
                match_vector = img_vector_dict[c1]
                for char, vector in img_vector_dict.items():
                    if char == c1:
                        continue
                    cosine_similar = cosine_similarity(match_vector, vector)
                    similarity_dict[char] = cosine_similar
                # 按相似度排序，取前10个
                sorted_similarity = sorted(similarity_dict.items(), key=itemgetter(1), reverse=True)
                print([(char, round(similarity, 4)) for char, similarity in sorted_similarity[:10]])
                save_li = [char for char, similarity in sorted_similarity[:10]]

                f.write(c1 + "\t" + " ".join(save_li) + "\n")

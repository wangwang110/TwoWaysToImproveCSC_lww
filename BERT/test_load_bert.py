import torch

# # bert_path = "/data_share/qiuwenbo/"
# bert_path = "/data_local/plm_models/bert_cased_L-12_H-768_A-12/"
# # bert_cased_L-12_H-768_A-12
# bert = torch.load(bert_path + "pytorch_model.bin")
# for i in bert:
#     print(i + '   ' + str(list(bert[i].size())))

# out = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [2, 2, 3, 4]], [[5, 6, 7, 8], [3, 2, 3, 4], [5, 6, 7, 8]]])
# # 2*3*4
# input_attn_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
# mask_soft = (1 - input_attn_mask) * (-1e4)
#
# outputs = out + mask_soft.unsqueeze(-1)
#
# print(outputs)


# torch.gather 按索引取数
x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [2, 2, 3, 4]])
print(x)

# 按照第二维度取数，indices第一个维度要和x一样
indices = torch.tensor([[1], [2], [3]])
# 意思是取[0,1],[2,2],[3,3]索引位置的数
# indices 是给的第二维的索引
print(torch.gather(x, 1, indices))

# 按照第一维度取数，indices第二个维度要和x一样
indices = torch.tensor([[1, 2, 1, 0]])
# 取[1,0],[2,1],[1,2],[0,3]索引位置的数
# indices 是给的第一维的索引
print(torch.gather(x, 0, indices))

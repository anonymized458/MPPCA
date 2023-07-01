import numpy as np
import copy

import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import mutual_info_score
from math import isnan
import pandas as pd

COMPRESSED_K_LEVELS = 256


# COMPRESS_BOUND = 10


def cal_freq_by_entropy_1(w_locals, lr, a, bound):
    return calculate_weit_by_entropy(w_locals, lr, a, bound)


# need client_cfl
# def cal_freq_by_entropy_2(clients):
#     return compute_frequency_cfl(clients)


'''
method 1
'''


# 传入单个客户端的模型参数列表并压缩
# 抄的
def uint_para_compress(delta_model_para, compress_k_levels, compress_bound, learning_rate):
    # note that we keep learning_rate here, as it may change over time
    if compress_k_levels < 2:
        return delta_model_para
    compressed_model = []
    k_level_delta = np.linspace(-compress_bound * learning_rate, compress_bound * learning_rate, compress_k_levels)
    dist = (2.0 * compress_bound * learning_rate) / (compress_k_levels - 1)
    for delta_model_index, para in enumerate(delta_model_para):
        para_shape = list(para.shape)
        if para.ndim > 1:
            para = para.flatten()
        # print(para)
        argmin_less = np.floor((para + compress_bound * learning_rate) / dist).astype(np.int32)
        argmin_larger = np.ceil((para + compress_bound * learning_rate) / dist).astype(np.int32)
        argmin_less[argmin_less < 0] = 0
        argmin_larger[argmin_larger < 0] = 0
        argmin_less[argmin_less > compress_k_levels - 1] = compress_k_levels - 1
        argmin_larger[argmin_larger > compress_k_levels - 1] = compress_k_levels - 1
        prop = (para - (k_level_delta[argmin_less])) / dist
        rannum = np.random.rand(len(para))
        int_array = np.where(rannum < prop, argmin_larger, argmin_less)
        if compress_k_levels <= 2 ** 8:
            int_array = int_array.astype(np.uint8)
        elif compress_k_levels <= 2 ** 16:
            int_array = int_array.astype(np.uint16)
        else:
            int_array = int_array.astype(np.uint32)
        int_array = int_array.reshape(para_shape)
        compressed_model.append(int_array)
    return compressed_model


# 传入从各个客户端模型直接获取的参数列表的列表：net.state_dict()数组
def compress_para(w_ori, lr, bound):
    w = copy.deepcopy(w_ori)
    compress_k_levels = COMPRESSED_K_LEVELS
    compress_bound = bound
    learning_rate = lr  # note it is 0.01
    compressed_w_list = []
    for i in range(len(w)):
        w_i = []
        for k in w[i].keys():
            w_i.append(w[i][k].cpu().numpy())
        compressed_w_i = uint_para_compress(w_i, compress_k_levels, compress_bound, learning_rate)
        compressed_w_list.append(compressed_w_i)
    return compressed_w_list


def calculate_weit_by_entropy(w, lr, a, bound):
    temp_para = copy.deepcopy(w)
    temp_para = compress_para(temp_para, lr, bound)
    HEIGHT = COMPRESSED_K_LEVELS
    user_num = len(temp_para)
    for user_index in range(len(temp_para)):
        for para_index in range(len(temp_para[0])):
            temp_para[user_index][para_index] = temp_para[user_index][para_index].flatten()
    temp_con_list = []
    for user_index in range(len(temp_para)):
        temp_con = copy.deepcopy(temp_para[user_index][0])
        for para_index in range(1, len(temp_para[0])):
            temp_con = np.concatenate((temp_con, temp_para[user_index][para_index]), axis=-1)
        temp_con_list.append(temp_con)
    con = copy.deepcopy(temp_con_list[0])
    con = np.expand_dims(con, axis=0)
    for user_index in range(1, len(temp_para)):
        temp_con_list[user_index] = np.expand_dims(temp_con_list[user_index], axis=0)
        con = np.concatenate((con, temp_con_list[user_index]), axis=0)
        # print('con.shape:')
        # print(con.shape)
    # sample_index =
    # # 获取到参数更新分布
    # possibility = np.zeros((user_num, HEIGHT))
    # para_num = len(con[0])
    # for i in range(user_num):
    #     for h in range(HEIGHT):
    #         possibility[i, h] = np.sum(con[i] == h)
    # possibility = possibility / para_num

    # con每一行是一个客户端的所有参数的离散更新值；每一列为一个客户端

    user_num = len(con)

    pairwise_entropy = np.zeros((user_num, user_num))
    entropy_i_list = np.zeros(user_num)
    for i in range(user_num):
        for j in range(user_num):
            # 计算互信息
            # TODO: 改成自己的
            # TODO: 为什么有的时候会出现他和任何人的互信息都是0
            pairwise_entropy[i][j] = mutual_info_score(con[i], con[j])

    # print('pairwise_entropy:')
    # print(pairwise_entropy)

    for i in range(user_num):
        entropy_i_list[i] = pairwise_entropy[i].sum()

    # 放大一下
    alpha = a
    entropy_i_list = np.exp(alpha * entropy_i_list)
    sum_entropy = entropy_i_list.sum()

    if sum_entropy == 0.0:
        return []

    entropy_i_list = entropy_i_list / sum_entropy

    print("entropy_i_list:")
    print(entropy_i_list)
    frequency = []
    for e in entropy_i_list:
        frequency.append(e)
    return frequency


# 计算两个模型之间的互信息
def cal_entropy_pairwise(client_w, cluster_w, lr, bound):
    temp_para = copy.deepcopy(client_w)
    # compress
    cw1 = []
    for k in temp_para.keys():
        cw1.append(temp_para[k].cpu().numpy())
    compressed_client_w = uint_para_compress(cw1, COMPRESSED_K_LEVELS, bound, lr)

    pass
    for para_index in range(len(compressed_client_w)):
        compressed_client_w[para_index] = compressed_client_w[para_index].flatten()
    con1 = copy.deepcopy(compressed_client_w[0])
    for para_index in range(1, len(compressed_client_w)):
        con1 = np.concatenate((con1, compressed_client_w[para_index]), axis=-1)

    temp_para = copy.deepcopy(cluster_w)
    cw2 = []
    for k in temp_para.keys():
        cw2.append(temp_para[k].cpu().numpy())
    # compress
    compressed_cluster_w = uint_para_compress(cw2, COMPRESSED_K_LEVELS, bound, lr)
    pass
    for para_index in range(len(compressed_cluster_w)):
        compressed_cluster_w[para_index] = compressed_cluster_w[para_index].flatten()
    con2 = copy.deepcopy(compressed_cluster_w[0])
    for para_index in range(1, len(compressed_cluster_w)):
        con2 = np.concatenate((con2, compressed_cluster_w[para_index]), axis=-1)

    entropy = mutual_info_score(con1, con2)
    return entropy


'''
method 2
'''

#
# def compute_frequency_cfl(clients):
#     return cal_pairwise_entropy([copy.deepcopy(client.dw) for client in clients])
#
#
# def flatten(source):
#     return torch.cat([value.flatten() for value in source.values()])
#
#
# # bug；不同客户端划分方式不一样
# def compress_para_by_cut(w_i, k_levels_delta):
#     # print('w_i:')
#     # print(w_i)
#     temp_w_i = pd.cut(w_i, bins=k_levels_delta, right=True, labels=False, duplicates='drop', precision=10)
#     compressed_w_i = []
#     for j in range(len(temp_w_i)):
#         if not isnan(temp_w_i[j]):
#             compressed_w_i.append(temp_w_i[j])
#     # print("compressed_w_i:")
#     # print(compressed_w_i)
#     return compressed_w_i
#
#
# def cal_pairwise_entropy(sources):
#     # print("sources:")
#     # print(sources)
#     compress_k_levels = 256
#     # angles = torch.zeros([len(sources), len(sources)])
#     flattened_sources = []
#     for i in range(len(sources)):
#         # sources[i] = flatten(sources[i])
#         flattened_sources.append(copy.deepcopy(flatten(sources[i])))
#     # for i, source1 in enumerate(sources):
#     #     for j, source2 in enumerate(sources):
#     #         s1 = flatten(source1)
#     #         s2 = flatten(source2)
#     #         angles[i, j] = torch.sum(s1 * s2) / (torch.norm(s1) * torch.norm(s2) + 1e-12)
#
#     compressed_w_list = []
#
#     for i in range(len(flattened_sources)):
#         flattened_sources[i] = flattened_sources[i].cpu().numpy()
#     # print("flattened:")
#     # print(flattened_sources)
#     max_para = np.max(flattened_sources)
#     min_para = np.min(flattened_sources)
#     k_levels_delta = np.linspace(min_para, max_para, compress_k_levels)
#     k_levels_delta[0] -= 1
#     print("max")
#     print(max_para)
#     print("min")
#     print(min_para)
#     print("delta")
#     print(k_levels_delta)
#
#     for i in range(len(flattened_sources)):
#         w_i = []
#         for k in range(len(flattened_sources[i])):
#             w_i.append(copy.deepcopy(flattened_sources[i][k]))
#         compressed_w_i = compress_para_by_cut(w_i, k_levels_delta)
#         compressed_w_list.append(compressed_w_i)
#
#     user_num = len(flattened_sources)
#     entropy = np.zeros((user_num, user_num))
#     entrpy_i_list = np.zeros(user_num)
#
#     for i in range(len(compressed_w_list)):
#         for j in range(len(compressed_w_list)):
#             # 计算互信息
#             # TODO: 改成自己的
#             # TODO: 为什么有的时候会出现他和任何人的互信息都是0
#             entropy[i][j] = mutual_info_score(compressed_w_list[i], compressed_w_list[j])
#
#     # print('pairwise_entropy:')
#     # print(entropy)
#
#     for i in range(user_num):
#         entrpy_i_list[i] = entropy[i].sum()
#
#     # 放大一下
#     alpha = 2
#     entrpy_i_list = np.exp(alpha * entrpy_i_list)
#     sum_entropy = entrpy_i_list.sum()
#
#     if sum_entropy == 0.0:
#         return []
#
#     entrpy_i_list = entrpy_i_list / sum_entropy
#     print("entrpy_i_list:")
#
#     print(entrpy_i_list)
#     frequency = []
#     for e in entrpy_i_list:
#         frequency.append(e)
#     return frequency

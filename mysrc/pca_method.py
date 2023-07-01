"""
基于PCA，对模型的每一层计算相似性和多样性，然后对所有层的相似性多样性基于参数数量聚合
"""
import numpy as np
import copy
import torch
from torch import nn, optim
import torch.nn.functional as F
import pandas as pd
import math


# 计算两两之间的相似性
# 传入所有客户端的模型参数（state_dict）列表
# state dict参数包含了所有层的权重w和偏差b，并且是分开作为两个键值对存储的
# 对于偏差的键值对，是一维数组，所以只分析权重矩阵
# 权重对应的不一定是二维数组（矩阵），所以要先把它转化为二维
# 对于一个高维矩阵，把最后一个维度对应的向量取出来，从一个(d1,d2,...,d-1)的矩阵变成(d1*d2*...*d-2,d-1)的二维矩阵
#

def cal_freq_by_layer_pca(w_locals, a):
    w = copy.deepcopy(w_locals)
    user_num = len(w)
    w_list = []
    for i in range(user_num):
        w_i = []
        for param_tensor in w[i].keys():
            # 只选择weight，抛弃bias
            if not param_tensor[-4:] == 'bias':
                # 先把torch转化为numpy
                layer_param = w[i][param_tensor].cpu().numpy()
                compressed_layer_param = high_array_2_double(layer_param)
                w_i.append(compressed_layer_param)
        # w_i包含每一层的二维数组（矩阵）形式的参数
        w_list.append(copy.deepcopy(w_i))

    similarity = np.zeros((user_num, user_num))
    for i in range(user_num):
        for j in range(user_num):
            similarity[i][j] = compare_pairwise(w_list[i], w_list[j])

    sim_list = np.zeros(user_num)
    for i in range(user_num):
        sim_list[i] = similarity[i].sum()

    alpha = a
    sim_list = np.exp(alpha * sim_list)
    sum_sim = sim_list.sum()
    sim_list = sim_list / sum_sim
    print(sim_list)
    freq = []
    for sim in sim_list:
        freq.append(sim)
    return freq


def high_array_2_double(ori_array):
    high = copy.deepcopy(ori_array)
    # low = high.reshape(-1, high.shape[-1])
    if len(high.shape) > 2:
        low = high.reshape(-1, high.shape[-1] * high.shape[-2])
    else:
        low = high.reshape(-1, high.shape[-1])
    return low


# 传入两个客户端的参数矩阵列表，比较客户端之间的相似度
def compare_pairwise(seller, buyer):
    layer_num = len(buyer)
    # 记录每一层的相似性和多样性
    rel_list = []
    div_list = []
    # param_num_list = []
    param_num_list = np.zeros(layer_num)
    # 每一层得到一组特征向量，和两组特征值，用这两组特征值计算相似度和多样性
    # 根据每一层的参数数量做聚合
    for i in range(layer_num):
        # 卖方的层参数
        buyer_layer = buyer[i]
        seller_layer = seller[i]
        param_num = len(buyer_layer) * len(buyer_layer[0])
        # param_num_list.append(copy.deepcopy(float(param_num)))
        param_num_list[i] = copy.deepcopy(float(param_num))
        # 计算相似性多样性
        rel_per, div_per = cal_single(buyer_layer, seller_layer)

        rel_list.append(copy.deepcopy(rel_per))
        div_list.append(copy.deepcopy(div_per))
    rel_freq = param_num_list / float(param_num_list.sum())
    rel = 0
    for i in range(layer_num):
        rel += rel_list[i] * rel_freq[i]
    return rel


def takeFirst(elem):
    return elem[0]


N_VEC = 5

from sklearn.decomposition import PCA


# 计算传入的两个（单层参数）矩阵的PCA相似度和多样性
def cal_single(buyer, seller):
    n_samples, n_features = buyer.shape
    n_vec = N_VEC
    # 计算买方协方差矩阵的特征值和特征向量
    buyer_mean = np.array([np.mean(buyer[:, i]) for i in range(n_features)])
    norm_buyer = buyer - buyer_mean
    buyer_scatter_matrix = np.dot(np.transpose(norm_buyer), norm_buyer)
    print("buyer_scatter_matrix:")
    print(buyer_scatter_matrix)
    buyer_val, buyer_vec = np.linalg.eig(buyer_scatter_matrix)
    # print("buyer_vec:")
    # print(buyer_vec)
    # print("buyer_val:")
    # print(buyer_val)
    buyer_pairs = [(np.abs(buyer_val[i]), buyer_vec[:, i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    buyer_pairs.sort(key=takeFirst, reverse=True)
    buyer_feature = np.array([ele[1] for ele in buyer_pairs[:n_vec]])
    buyer_feature2 = np.zeros(buyer_feature.shape)

    for i in range(len(buyer_feature)):
        for j in range(len(buyer_feature[0])):
            buyer_feature2[i][j] = copy.deepcopy(buyer_feature[i][j].real)
            # print(buyer_feature[i][j].real)

    # print("buyer_feature:")
    # print(buyer_feature2)
    # 计算卖方协方差矩阵在买方特征向量上的二阶矩（特征值）
    seller_mean = np.array([np.mean(seller[:, i]) for i in range(n_features)])
    norm_seller = seller - seller_mean
    seller_scatter_matrix = np.dot(np.transpose(norm_seller), norm_seller)
    # print("buyer_feature:")
    # print(buyer_feature)

    seller_val = []
    for i in range(n_vec):
        data = np.dot(seller_scatter_matrix, np.transpose(buyer_feature[i]))
        # print("data:")
        # print(data)
        second_moment = 0
        for j in range(len(data)):
            second_moment += data[j] * data[j]
        second_moment = math.sqrt(second_moment)
        seller_val.append(copy.deepcopy(second_moment))
    div = 1.0
    rel = 1.0
    denominator = 1.0
    for j in range(n_vec):
        rel = rel * min(buyer_val[j], seller_val[j])
        div = div * abs(buyer_val[j] - seller_val[j])
        denominator = denominator * max(buyer_val[j], seller_val[j])
    rel = rel / denominator
    div = div / denominator
    d = 1.0 / n_features
    rel = rel ** d
    div = div ** d
    return rel, div


def cal_single_by_sklearn(buyer, seller):
    n_samples, n_features = buyer.shape
    n_vec = N_VEC
    # sklearn.PCA
    pca = PCA(n_components=5)
    pca.fit(buyer)
    buyer_mean = np.array([np.mean(buyer[:, i]) for i in range(n_features)])
    norm_buyer = buyer - buyer_mean
    buyer_scatter_matrix = np.dot(np.transpose(norm_buyer), norm_buyer)
    seller_mean = np.array([np.mean(seller[:, i]) for i in range(n_features)])
    norm_seller = seller - seller_mean
    seller_scatter_matrix = np.dot(np.transpose(norm_seller), norm_seller)
    compressed_seller = pca.transform(seller_scatter_matrix)
    compressed_buyer = pca.transform(buyer_scatter_matrix)
    seller_val = []
    buyer_val = []
    for i in range(len(compressed_buyer)):
        second_moment = 0
        for j in range(len(compressed_buyer[i])):
            second_moment += compressed_buyer[i][j] * compressed_buyer[i][j]
        second_moment = math.sqrt(second_moment)
        buyer_val.append(copy.deepcopy(second_moment))
    for i in range(len(compressed_seller)):
        second_moment = 0
        for j in range(len(compressed_seller[i])):
            second_moment += compressed_seller[i][j] * compressed_seller[i][j]
        second_moment = math.sqrt(second_moment)
        seller_val.append(copy.deepcopy(second_moment))
    div = 1.0
    rel = 1.0
    denominator = 1.0
    for j in range(n_vec):
        rel = rel * min(buyer_val[j], seller_val[j])
        div = div * abs(buyer_val[j] - seller_val[j])
        denominator = denominator * max(buyer_val[j], seller_val[j])
    rel = rel / denominator
    div = div / denominator
    d = 1.0 / n_features
    rel = rel ** d
    div = div ** d
    return rel, div

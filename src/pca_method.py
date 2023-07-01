"""
基于PCA，对模型的每一层计算相似性和多样性，然后对所有层的相似性多样性基于参数数量聚合
"""
import numpy as np
import cupy as cp
import copy
import torch
from torch import nn, optim
import torch.nn.functional as F
import pandas as pd
import math
from sklearn.decomposition import PCA


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
            if param_tensor == 'fc3.weight':
                layer_param = w[i][param_tensor].cpu().numpy()
                compressed_layer_param = high_array_2_double(layer_param)
                w_i.append(cp.asarray(compressed_layer_param))
            # if not param_tensor[-4:] == 'bias':
            #     # 先把torch转化为numpy
            #
            #     layer_param = w[i][param_tensor].cpu().numpy()
            #     compressed_layer_param = high_array_2_double(layer_param)
            #     w_i.append(compressed_layer_param)
        # w_i包含每一层的二维数组（矩阵）形式的参数
        w_list.append(copy.deepcopy(w_i))

    similarity = cp.zeros((user_num, user_num))
    for i in range(user_num):
        for j in range(user_num):
            similarity[i][j] = compare_pairwise(w_list[i], w_list[j])

    sim_list = cp.zeros(user_num)
    for i in range(user_num):
        sim_list[i] = similarity[i].sum()

    alpha = a
    sim_list = cp.exp(alpha * sim_list)
    sum_sim = sim_list.sum()
    sim_list = sim_list / sum_sim
    print(sim_list)
    freq = []
    for sim in sim_list:
        freq.append(cp.asnumpy(sim))
    return cp.asnumpy(freq)


def cal_freq_by_layer_pca_complex(w_locals, a, sigma):
    w = copy.deepcopy(w_locals)
    user_num = len(w)
    w_list = []
    for i in range(user_num):
        w_i = []
        for param_tensor in w[i].keys():
            # 只选择weight，抛弃bias
            if param_tensor == 'fc3.weight':
                layer_param = w[i][param_tensor].cpu().numpy()
                compressed_layer_param = high_array_2_double(layer_param)
                w_i.append(cp.asarray(compressed_layer_param))
            # if not param_tensor[-4:] == 'bias':
            #     # 先把torch转化为numpy
            #
            #     layer_param = w[i][param_tensor].cpu().numpy()
            #     compressed_layer_param = high_array_2_double(layer_param)
            #     w_i.append(compressed_layer_param)
        # w_i包含每一层的二维数组（矩阵）形式的参数
        w_list.append(copy.deepcopy(w_i))

    similarity = cp.zeros((user_num, user_num))
    diversity = cp.zeros((user_num, user_num))
    for i in range(user_num):
        for j in range(user_num):
            similarity[i][j], diversity[i][j] = compare_pairwise_complex(w_list[i], w_list[j])

    sim_list = cp.zeros(user_num)
    div_list = cp.zeros(user_num)
    for i in range(user_num):
        sim_list[i] = similarity[i].sum()
        div_list[i] = diversity[i].sum()

    alpha = a
    sim_list = cp.exp(alpha * sim_list)
    div_list = cp.exp(alpha * div_list)
    sum_sim = sim_list.sum()
    sum_div = div_list.sum()
    sim_list = sim_list / sum_sim
    div_list = div_list / sum_div
    # print(sim_list)
    tradeoff = sigma
    freq = []
    for i in range(len(sim_list)):
        freq.append(cp.asnumpy(sim_list[i] * tradeoff + div_list[i] * (1 - tradeoff)))
    return cp.asnumpy(freq)


def cal_div_freq_by_layer_pca(w_locals, a):
    w = copy.deepcopy(w_locals)
    user_num = len(w)
    w_list = []
    for i in range(user_num):
        w_i = []
        for param_tensor in w[i].keys():
            # 只选择weight，抛弃bias
            if param_tensor == 'fc3.weight':
                layer_param = w[i][param_tensor].cpu().numpy()
                compressed_layer_param = high_array_2_double(layer_param)
                w_i.append(cp.asarray(compressed_layer_param))
            # if not param_tensor[-4:] == 'bias':
            #     # 先把torch转化为numpy
            #
            #     layer_param = w[i][param_tensor].cpu().numpy()
            #     compressed_layer_param = high_array_2_double(layer_param)
            #     w_i.append(compressed_layer_param)
        # w_i包含每一层的二维数组（矩阵）形式的参数
        w_list.append(copy.deepcopy(w_i))

    diversity = cp.zeros((user_num, user_num))
    for i in range(user_num):
        for j in range(user_num):
            diversity[i][j] = compare_div(w_list[i], w_list[j])

    div_list = cp.zeros(user_num)
    for i in range(user_num):
        div_list[i] = diversity[i].sum()

    alpha = a
    div_list = cp.exp(alpha * div_list)
    sum_div = div_list.sum()
    div_list = div_list / sum_div
    print(div_list)
    freq = []
    for div in div_list:
        freq.append(cp.asnumpy(div))
    return cp.asnumpy(freq)


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
    param_num_list = cp.zeros(layer_num)
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


def compare_pairwise_complex(seller, buyer):
    layer_num = len(buyer)
    # 记录每一层的相似性和多样性
    rel_list = []
    div_list = []
    # param_num_list = []
    param_num_list = cp.zeros(layer_num)
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
    div = 0
    for i in range(layer_num):
        rel += rel_list[i] * rel_freq[i]
        div += div_list[i] * rel_freq[i]
    return rel, div


def compare_div(seller, buyer):
    layer_num = len(buyer)
    # 记录每一层的相似性和多样性
    rel_list = []
    div_list = []
    # param_num_list = []
    param_num_list = cp.zeros(layer_num)
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
    div_freq = param_num_list / float(param_num_list.sum())
    div = 0
    for i in range(layer_num):
        div += div_list[i] * div_freq[i]
    return div


def takeFirst(elem):
    return elem[0]


N_VEC = 3


# 计算传入的两个（单层参数）矩阵的PCA相似度和多样性
def cal_single(buyer, seller):
    buyer = cp.asarray(buyer)
    seller = cp.asarray(seller)
    n_samples, n_features = buyer.shape
    n_vec = N_VEC
    # 计算买方协方差矩阵的特征值和特征向量
    buyer_mean = cp.array([cp.mean(buyer[:, i]) for i in range(n_features)])
    norm_buyer = buyer - buyer_mean
    buyer_scatter_matrix = cp.dot(cp.transpose(norm_buyer), norm_buyer)
    buyer_scatter_matrix = cp.asnumpy(buyer_scatter_matrix)
    buyer_val, buyer_vec = np.linalg.eig(buyer_scatter_matrix)
    # buyer_scatter_matrix = cp.asarray(buyer_scatter_matrix)
    buyer_pairs = [(cp.abs(buyer_val[i]), buyer_vec[:, i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    buyer_pairs.sort(key=takeFirst, reverse=True)
    buyer_feature = cp.array([ele[1] for ele in buyer_pairs[:n_vec]])
    buyer_feature2 = cp.zeros(buyer_feature.shape)

    for i in range(len(buyer_feature)):
        for j in range(len(buyer_feature[0])):
            buyer_feature2[i][j] = copy.deepcopy(buyer_feature[i][j].real)
    # 计算卖方协方差矩阵在买方特征向量上的二阶矩（特征值）
    seller_mean = cp.array([cp.mean(seller[:, i]) for i in range(n_features)])
    norm_seller = seller - seller_mean
    seller_scatter_matrix = cp.dot(cp.transpose(norm_seller), norm_seller)
    # seller_scatter_matrix = cp.asarray(seller_scatter_matrix)
    seller_val = []
    for i in range(n_vec):
        data = cp.dot(seller_scatter_matrix, cp.transpose(buyer_feature2[i]))
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
    rel = (np.abs(rel)) ** d
    div = div ** d
    return rel, div


def cal_freq_by_simple_pca(w_locals, a):
    w = copy.deepcopy(w_locals)
    user_num = len(w)
    w_list = []
    for i in range(user_num):
        w_i = []
        for param_tensor in w[i].keys():
            # 只选择weight，抛弃bias
            if param_tensor == 'fc3.weight':
                layer_param = w[i][param_tensor].cpu().numpy()
                compressed_layer_param = high_array_2_double(layer_param)
                w_i.append(cp.asarray(compressed_layer_param))
            # if not param_tensor[-4:] == 'bias':
            #     # 先把torch转化为numpy
            #
            #     layer_param = w[i][param_tensor].cpu().numpy()
            #     compressed_layer_param = high_array_2_double(layer_param)
            #     w_i.append(compressed_layer_param)
        # w_i包含每一层的二维数组（矩阵）形式的参数
        w_list.append(copy.deepcopy(w_i))

    diversity = cp.zeros((user_num, user_num))
    for i in range(user_num):
        for j in range(user_num):
            diversity[i][j] = compare_pairwise_simple(w_list[i], w_list[j])

    div_list = cp.zeros(user_num)
    for i in range(user_num):
        div_list[i] = diversity[i].sum()

    # alpha需要是负数，这样才能基于差异性度量按关联性计算权重
    alpha = a
    div_list = cp.exp(alpha * div_list)
    sum_div = div_list.sum()
    div_list = div_list / sum_div
    print(div_list)
    freq = []
    for div in div_list:
        freq.append(cp.asnumpy(div))
    return cp.asnumpy(freq)


def compare_pairwise_simple(seller, buyer):
    layer_num = len(buyer)
    # 记录每一层的相似性和多样性
    rel_list = []
    div_list = []
    # param_num_list = []
    param_num_list = cp.zeros(layer_num)
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
        div_per = cal_single_simple(buyer_layer, seller_layer)

        # rel_list.append(copy.deepcopy(rel_per))
        div_list.append(copy.deepcopy(div_per))
    div_freq = param_num_list / float(param_num_list.sum())
    div = 0
    for i in range(layer_num):
        div += div_list[i] * div_freq[i]
    return div


# 计算欧式距离，是差异性的度量
def cal_single_simple(buyer, seller):
    buyer = cp.asnumpy(buyer)
    seller = cp.asnumpy(seller)
    pca1 = PCA(n_components=3)
    pca1.fit(buyer)
    compressed_buyer = pca1.transform(buyer)
    pca2 = PCA(n_components=3)
    pca2.fit(seller)
    compressed_seller = pca2.transform(seller)
    dis = 0
    for i in range(len(compressed_buyer)):
        for j in range(len(compressed_buyer[0])):
            dis += math.pow((compressed_seller[i][j] - compressed_buyer[i][j]), 2)
    dis = math.sqrt(dis)
    return dis

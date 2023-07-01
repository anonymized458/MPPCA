import copy
import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy


def cal_freq_by_cos(w_locals, a):
    w = copy.deepcopy(w_locals)
    w_list = []
    # angles = np.zeros((len(w), len(w)))

    for i in range(len(w)):
        w_i = []
        for k in w[i].keys():
            w_i.append(w[i][k].cpu().numpy())
        w_list.append(w_i)

    user_num = len(w_list)

    for user_index in range(len(w_list)):
        for para_index in range(len(w[0])):
            w_list[user_index][para_index] = w_list[user_index][para_index].flatten()
    temp_con_list = []
    for user_index in range(len(w_list)):
        temp_con = copy.deepcopy(w_list[user_index][0])
        for para_index in range(1, len(w_list[0])):
            temp_con = np.concatenate((temp_con, w_list[user_index][para_index]), axis=-1)
        temp_con_list.append(temp_con)
    con = copy.deepcopy(temp_con_list[0])
    con = np.expand_dims(con, axis=0)
    for user_index in range(1, len(w_list)):
        temp_con_list[user_index] = np.expand_dims(temp_con_list[user_index], axis=0)
        con = np.concatenate((con, temp_con_list[user_index]), axis=0)

    # for i in range(user_num):
    #     for j in range(user_num):
    #         angles[i][j] = cosine_similarity(con[i], con[j])
    angles = cosine_similarity(con)

    sim_list = np.zeros(len(w_list))
    for i in range(len(w_list)):
        sim_list[i] = angles[i].sum()

    alpha = a
    sim_list = np.exp(alpha * sim_list)
    sum_sim = sim_list.sum()
    sim_list = sim_list / sum_sim
    print(sim_list)
    freq = []
    for sim in sim_list:
        freq.append(sim)
    return freq


def compare_cos_pairwise(client_w, cluster_w):
    temp_para = copy.deepcopy(client_w)
    # compress
    cw1 = []
    for k in temp_para.keys():
        cw1.append(temp_para[k].cpu().numpy())
    # compressed_client_w = uint_para_compress(cw1, COMPRESSED_K_LEVELS, bound, lr)

    pass
    for para_index in range(len(cw1)):
        cw1[para_index] = cw1[para_index].flatten()
    con1 = copy.deepcopy(cw1[0])
    for para_index in range(1, len(cw1)):
        con1 = np.concatenate((con1, cw1[para_index]), axis=-1)

    temp_para = copy.deepcopy(cluster_w)
    cw2 = []
    for k in temp_para.keys():
        cw2.append(temp_para[k].cpu().numpy())
    # compress
    # compressed_cluster_w = uint_para_compress(cw2, COMPRESSED_K_LEVELS, bound, lr)
    pass
    for para_index in range(len(cw2)):
        cw2[para_index] = cw2[para_index].flatten()
    con2 = copy.deepcopy(cw2[0])
    for para_index in range(1, len(cw2)):
        con2 = np.concatenate((con2, cw2[para_index]), axis=-1)
    con1 = con1.reshape(1,-1)
    con2 = con2.reshape(1, -1)
    similarity = cosine_similarity(con1, con2)
    return similarity


def cal_freq_by_l2(w_locals, a):
    w = copy.deepcopy(w_locals)
    w_list = []

    for i in range(len(w)):
        w_i = []
        for k in w[i].keys():
            w_i.append(w[i][k].cpu().numpy())
        w_list.append(w_i)

    user_num = len(w_list)

    for user_index in range(len(w_list)):
        for para_index in range(len(w[0])):
            w_list[user_index][para_index] = w_list[user_index][para_index].flatten()
    temp_con_list = []
    for user_index in range(len(w_list)):
        temp_con = copy.deepcopy(w_list[user_index][0])
        for para_index in range(1, len(w_list[0])):
            temp_con = np.concatenate((temp_con, w_list[user_index][para_index]), axis=-1)
        temp_con_list.append(temp_con)
    con = copy.deepcopy(temp_con_list[0])
    con = np.expand_dims(con, axis=0)
    for user_index in range(1, len(w_list)):
        temp_con_list[user_index] = np.expand_dims(temp_con_list[user_index], axis=0)
        con = np.concatenate((con, temp_con_list[user_index]), axis=0)

    l2 = np.zeros((user_num, user_num))

    for i in range(user_num):
        for j in range(user_num):
            l2[i][j] = np.linalg.norm(con[i] - con[j])

    l2_list = np.zeros(len(w_list))
    for i in range(len(w_list)):
        l2_list[i] = l2[i].sum()

    alpha = a
    l2_list = np.exp(alpha * l2_list)
    sum_l2 = l2_list.sum()
    l2_list = l2_list / sum_l2
    print(l2_list)
    freq = []
    for l in l2_list:
        freq.append(l)
    return freq


# def consine_similarity

COMPRESSED_K_LEVELS = 256
COMPRESS_BOUND = 10


def cal_freq_by_kl(w, lr, a):
    temp_para = copy.deepcopy(w)
    temp_para = compress_para(temp_para, lr)
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

    pos = np.zeros((user_num, HEIGHT))
    for i in range(user_num):
        for h in range(HEIGHT):
            pos[i, h] = np.sum(con[i] == h)

    para_size = len(con[0])
    # 概率分布
    pos = pos / para_size

    pairwise_entropy = np.zeros((user_num, user_num))
    entropy_i_list = np.zeros(user_num)
    for i in range(user_num):
        for j in range(user_num):
            # KL散度
            pairwise_entropy[i][j] = entropy(pos[i], pos[j])

    # print('pairwise_entropy:')
    # print(pairwise_entropy)

    # KL散度范围（0，∞），越小越相似：映射到一个递减趋近于0的恒大于0的函数
    for i in range(user_num):
        entropy_i_list[i] = pairwise_entropy[i].sum()

    # 放大一下
    alpha = a
    entropy_i_list = np.pi - np.arctan(alpha * entropy_i_list)
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
def compress_para(w_ori, lr):
    w = copy.deepcopy(w_ori)
    compress_k_levels = COMPRESSED_K_LEVELS
    compress_bound = COMPRESS_BOUND
    learning_rate = lr  # note it is 0.01
    compressed_w_list = []
    for i in range(len(w)):
        w_i = []
        for k in w[i].keys():
            w_i.append(w[i][k].cpu().numpy())
        compressed_w_i = uint_para_compress(w_i, compress_k_levels, compress_bound, learning_rate)
        compressed_w_list.append(compressed_w_i)
    return compressed_w_list

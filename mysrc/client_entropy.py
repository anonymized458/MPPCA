import numpy as np
import copy

import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import mutual_info_score


def mycopy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


# 传入单个客户端的模型参数列表并压缩
# 抄的
# TODO: 有啥用
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
        # noise = 1
        # para = para + noise * np.random.normal(0, 1, para.shape)
        #        print ((para+compress_bound * learning_rate)/dist)
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


COMPRESSED_K_LEVELS = 8


# 传入从各个客户端模型直接获取的参数列表的列表：net.state_dict()数组
# TODO: recover是干什么的
def compress_para(w, lr):
    compressed_w_list = []
    compress_k_levels = COMPRESSED_K_LEVELS
    compress_bound = 10
    learning_rate = lr  # note it is 0.01
    for i in range(len(w)):
        w_i = []
        for k in w[0].keys():
            w_i.append(w[i][k].cpu().numpy())
        compressed_w_i = uint_para_compress(w_i, compress_k_levels, compress_bound, learning_rate)
        compressed_w_list.append(compressed_w_i)
    return compressed_w_list


# 传入压缩后的客户端参数列表，返回聚合权重
def calculate_weit_by_entropy(temp_para):
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

    pairwise_entropy = np.zeros((user_num, user_num))
    entrpy_i_list = np.zeros(user_num)
    for i in range(user_num):
        for j in range(user_num):
            # 计算互信息
            # TODO: 改成自己的
            pairwise_entropy[i][j] = mutual_info_score(con[i], con[j])
    for i in range(user_num):
        entrpy_i_list[i] = np.sum(pairwise_entropy[i])
    sum_entropy = np.sum(entrpy_i_list)
    entrpy_i_list = entrpy_i_list / sum_entropy
    frequency = []
    for e in entrpy_i_list:
        frequency.append(e)
    return frequency


class Client_Entropy(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device,
                 train_dl_local=None, test_dl_local=None):

        self.name = name
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr
        self.momentum = momentum
        self.device = device
        self.net.to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = train_dl_local
        self.ldr_test = test_dl_local
        self.acc_best = 0
        self.count = 0
        self.save_best = True
        self.w = {key: value for key, value in self.net.named_parameters()}
        self.w_old = {key: torch.zeros_like(value) for key, value in self.net.named_parameters()}
        self.dw = {key: torch.zeros_like(value) for key, value in self.net.named_parameters()}

    def train(self, is_print=False):
        self.net.to(self.device)
        self.net.train()
        mycopy(target=self.w_old, source=self.w)

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)

        epoch_loss = []
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                # optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        #         if self.save_best:
        #             _, acc = self.eval_test()
        #             if acc > self.acc_best:
        #                 self.acc_best = acc
        subtract_(target=self.dw, minuend=self.w, subtrahend=self.w_old)
        mycopy(target=self.w, source=self.w_old)
        return sum(epoch_loss) / len(epoch_loss)

    def get_state_dict(self):
        return self.net.state_dict()

    def get_best_acc(self):
        return self.acc_best

    def get_count(self):
        return self.count

    def get_net(self):
        return self.net

    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)

    def eval_test(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy

    def eval_train(self):
        self.net.to(self.device)
        self.net.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy

import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn.metrics import mutual_info_score
import copy


def mycopy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


from math import isnan


# bug；不同客户端划分方式不一样
def compress_para(w_i, k_levels_delta):
    print('w_i:')
    print(w_i)
    temp_w_i = pd.cut(w_i, bins=k_levels_delta, right=True, labels=False, duplicates='drop', precision=10)
    compressed_w_i = []
    for j in range(len(temp_w_i)):
        if not isnan(temp_w_i[j]):
            compressed_w_i.append(temp_w_i[j])
    print("compressed_w_i:")
    print(compressed_w_i)
    return compressed_w_i


def pairwise_entropy(sources):
    # print("sources:")
    # print(sources)
    compress_k_levels = 256
    # angles = torch.zeros([len(sources), len(sources)])
    flattened_sources = []
    for i in range(len(sources)):
        # sources[i] = flatten(sources[i])
        flattened_sources.append(copy.deepcopy(flatten(sources[i])))
    # for i, source1 in enumerate(sources):
    #     for j, source2 in enumerate(sources):
    #         s1 = flatten(source1)
    #         s2 = flatten(source2)
    #         angles[i, j] = torch.sum(s1 * s2) / (torch.norm(s1) * torch.norm(s2) + 1e-12)

    compressed_w_list = []

    for i in range(len(flattened_sources)):
        flattened_sources[i] = flattened_sources[i].cpu().numpy()
    # print("flattened:")
    # print(flattened_sources)
    max_para = np.max(flattened_sources)
    min_para = np.min(flattened_sources)
    k_levels_delta = np.linspace(min_para, max_para, compress_k_levels)
    k_levels_delta[0] -= 1
    print("max")
    print(max_para)
    print("min")
    print(min_para)
    print("delta")
    print(k_levels_delta)

    for i in range(len(flattened_sources)):
        w_i = []
        for k in range(len(flattened_sources[i])):
            w_i.append(copy.deepcopy(flattened_sources[i][k]))
        compressed_w_i = compress_para(w_i, k_levels_delta)
        compressed_w_list.append(compressed_w_i)

    user_num = len(flattened_sources)
    entropy = np.zeros((user_num, user_num))
    entrpy_i_list = np.zeros(user_num)

    for i in range(len(compressed_w_list)):
        for j in range(len(compressed_w_list)):
            # 计算互信息
            # TODO: 改成自己的
            # TODO: 为什么有的时候会出现他和任何人的互信息都是0
            entropy[i][j] = mutual_info_score(compressed_w_list[i], compressed_w_list[j])

    print('pairwise_entropy:')
    print(entropy)

    for i in range(user_num):
        entrpy_i_list[i] = entropy[i].sum()

    # 放大一下
    alpha = 10
    entrpy_i_list = np.exp(alpha * entrpy_i_list)
    sum_entropy = entrpy_i_list.sum()

    if sum_entropy == 0.0:
        return []

    entrpy_i_list = entrpy_i_list / sum_entropy
    # print

    print(entrpy_i_list)
    frequency = []
    for e in entrpy_i_list:
        frequency.append(e)
    return frequency


def compute_frequency_cfl(clients):
    return pairwise_entropy([copy.deepcopy(client.dw) for client in clients])


# def cluster_clients(S):
#     clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)
#
#     c1 = np.argwhere(clustering.labels_ == 0).flatten()
#     c2 = np.argwhere(clustering.labels_ == 1).flatten()
#     return c1, c2


def compute_max_update_norm(cluster):
    return np.max([torch.norm(flatten(client.dw)).item() for client in cluster])


def compute_mean_update_norm(cluster):
    return torch.norm(torch.mean(torch.stack([flatten(client.dw) for client in cluster]),
                                 dim=0)).item()


class Client_Entropy_CFL(object):
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
        return self.net.state_dict(), sum(epoch_loss) / len(epoch_loss)

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

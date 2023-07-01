import copy
import torch
from torch import nn
import numpy as np


def FedAvg(w, weight_avg=None):
    """
    Federated averaging
    :param w: list of client model parameters
    :return: updated server model parameters
    """
    if weight_avg is None:
        weight_avg = [1 / len(w) for i in range(len(w))]

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k].cuda() * weight_avg[0]

    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k].cuda() + w[i][k].cuda() * weight_avg[i]
        # w_avg[k] = torch.div(w_avg[k].cuda(), len(w))
    return w_avg


# def FedAvgWithBadUser(w, weight_avg=None):
#     bad_user_num = 3
#     if weight_avg is None:
#         weight_avg = [1 / len(w) for i in range(len(w))]
#
#     w2 = []
#
#     for i in range(len(w)):
#         wi = {}
#         for k in w[0].keys():
#             if i < bad_user_num:
#                 wi[k] = torch.from_numpy(0.01 * np.random.normal(0, 1, w[i][k].cpu().numpy().shape))
#             else:
#                 wi[k] = w[i][k]
#         w2.append(copy.deepcopy(wi))
#
#     w_avg = copy.deepcopy(w2[0])
#     for k in w_avg.keys():
#         w_avg[k] = w_avg[k].cuda() * weight_avg[0]
#
#     for k in w_avg.keys():
#         for i in range(1, len(w2)):
#             temp1 = w_avg[k]
#             temp2 = w2[i][k]
#             w_avg[k] = temp1.cuda() + temp2.cuda() * weight_avg[i]
#         # w_avg[k] = torch.div(w_avg[k].cuda(), len(w))
#     return w_avg

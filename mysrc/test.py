# import numpy as np
# import random
# #
# # # con = np.zeros((2, 4))
# # # for i in range(2):
# # #     for j in range(4):
# # #         con[i, j] = i * 4 + j
# # # # TB = TA[:, 3]
# # # # print(TA)
# # # # print(TA[1, 2])
# # # # print(TB)
# # #
# # # TotLen = len(con[0])
# # # temp_rand = list(range(TotLen))
# # # # random.shuffle(temp_rand)
# # # index_A = temp_rand[:int(TotLen / 2)]
# # # index_B = temp_rand[int(TotLen / 2):]
# # #
# # # para_A = con[:, index_A]
# # # para_B = con[:, index_B]
# # # len_A = len(para_A[0])
# # # len_B = len(para_B[0])
# # # TA = np.zeros((2, 4))
# # #
# # # print(con)
# # # print(temp_rand)
# # # print(index_A)
# # # print(index_B)
# # # print(para_A)
# # # print(para_B)
# # # TA[0, 1] = np.sum(para_A[0] == 1)
# # # TA[1, 3] = np.sum(para_A[1] == 4)
# # # print(TA)
# # # fre = []
# # # sum = np.zeros(2)
# # # for i in range(2):
# # #     # sum[i] = np.sum(con[i])
# # #     fre.append(con[0][i])
# # #
# # # # print(sum / np.sum(sum))
# # # print(fre)
# # #
# # # # [1] 38889
# # # # recovered list 什么意思
# # # # 从某一次开始，出现一个客户端和其他客户端的信息都是0；然后下一次开始所有互信息都变成0
# # # # 好像是因为压缩时的传入参数变成nan了
# # # import pandas as pd
# # # ages = np.array([1,5,10,40,36,12,58,62,77,89,100,18,20,25,30,32]) #年龄数据
# # # out = pd.cut(ages, [0,5,20,30,50,100], labels=[u"婴儿",u"青年",u"中年",u"壮年",u"老年"])
# # # print(out)
# # # a = []
# # # for i in range(16):
# # #     a.append(out[i])
# # # print(a)
# # # a = [1,1,2,3,4,5,5,6,6,6,6,7,7,8]
# # # print(a.count(1))
# #
# #
# # '''
# # 1.在一个轮次中，全局模型和训练后的本地模型一样
# # 2.训练后的模型参数正常，压缩时得到的参数nan
# # 3.compressed_w_i中有一行是浮点数，中间有一个nan /删掉浮点数了
# # 4.怎么改成float64
# # 5.训练loss变成nan
# #
# #
# #
# # 全局模型不变了，肯定是本地模型训练前后没有变化，所以聚合得到的全局模型不变
# # 训练方法肯定是正常的，所以是下发的全局模型导致的，训练前后没有变化
# # - 好像是因为读取模型参数 / 把那个部分删掉就好了
# #
# # 另外一个问题好像是学习率导致的，把学习率降低到0.001就好了，但是好像全局模型的loss和准确度抖动很大
# # 明明本地准确度挺高的
# # 可能的问题
# #
# # 1.alpha值太大的了
# # 2.数据异质性设计问题
# #
# # 2.17:17
# # ifca
# # fedavg1 a=2 k=128 35993
# # fedavg1 a=2 k=64 43983
# # fedavg0 fmnist 17053
# # fedavg1 fmnist a=2 k=256 19637
# # fedavg0 cifar100 52567
# # mypca cifar100 57545
# # 2.19:16
# # fedavg cifar100 resnet9
# # mypca a=0.5 k=256 cifar100 resnet9
# # fedavg fmnist
# # mypca a=0.5 k=256 fmnist
# # mypca a=0.5 k=256 cifar10
# # mypca a=2 k=256 fmnist
# # fedavg noniid3
# # mypca noniid3
# # 2.20
# # 0-3个随机标签，但是每个标签数量固定
# # noniid1分区方法
# # 0-3个随机标签，客户端数据量固定
# # 15：00
# # pca replaced
# # noniid5 *2
# # noniid7 *2
# # 加噪声拿n=1 *2
# # 迪利克雷倾斜
# #
# # 2.21
# # 对参数更新计算信息熵
# # replaced noniid label3 bound=10
# #
# #
# # 效果不好有没有可能是因为，每个客户端的数据都是平均分来的两个，所以
# #
# #
# # 2.22
# # 针对参数更新计算互信息
# # unreplaced label2
# # replaced label3 + bound=10
# # replaced label5
# # replaced label7
# #
# # replaced label3 + bound=5
# # replaced label3 + bound=2
# # replaced label3 + bound=1
# #
# # unreplaced label2 + bound=0.1
# # unreplaced label2 + bound=0.1 + a=1
# # unreplaced label2 + bound=0.1 + a=5
# # unreplaced label2 + bound=0.1 + a=0.5
# # unreplaced label2 + bound=0.1 + a=0.5 + k=128
# # k=64
# # k=32
# # k=16
# #
# # bound=0.01
# # bound=0.02
# #
# # bound=50
# # bound=100
# # bound=20
# #
# # bound=1 k=1024
# #
# # 模型互信息 bound=100
# #
# # bound=10 a=1
# #
# #
# # 打印数据 e-2~e-4
# # 为什么改变bound会改变方差
# # 如果是极端情况，过大或者过小的bound，所有更新被离散化到相同位置，所有人的互信息就会差不多，方差小
# # 所以说，其他情况不变，方差比较大的bound设置，说明此时的互信息是
# # '''
# #
# # # dict = {"1.weight": [1], "1.bias": [2], "2.weight": [3], "2.bias": [4]}
# # # print(dict)
# # # for w in dict.keys():
# # #     # print(w[-4:])
# # #     if w[-4:] == 'bias':
# # #         print(w)
# #
# # # a = np.zeros(6 * 3 * 5 * 5)
# # # for i in range(6 * 3 * 5 * 5):
# # #     # print(i)
# # #     a[i] = i % 25
# # # # print(a)
# # # a = a.reshape((6, 3, 5, 5))
# # # # print(a)
# # # if len(a.shape) > 2:
# # #     # print('yes')
# # #     print(a.reshape(-1, a.shape[-1] * a.shape[-2]))
# # '''
# # mypca update bound=10,5,2,1,20,50
# # cos param alpha=1,2,5,10,20,50,100
# # entropy cluster icfa n=2
# #
# # mypca update bound=0.5,0.1
# # mypca update bound=1 a=1,0.5,5
# #
# # 3.1
# # 可重复标签，每个客户端数据量一致:label5
# #     fedavg基准
# #     测试模型参数更新计算互信息时alpha的影响（bound=0.2）0.1~20////感觉可以用别的再测测，10
# #     基于模型的pca alpha的影响 0.1~2 ////错了 main文件写成PCA了
# #     基于参数更新的余弦相似度 alpha=1~50
# #
# #     测试模型参数更新计算互信息时alpha的影响（bound=10）0.5~10
# #
# #
# # 3.2
# # 同上
# #     测试模型参数更新计算PCA时alpha的影响0.1~10
# #     余弦相似度a=0.1~0.8
# #     模型参数更新 bound=1时a=0.1~10 全都写成bound=10了
# #     l2a=0.2~10 xxx
# #     基于互信息的分簇n=2 bound=0.5~10
# # '''
# #
# import numpy as np
# from pca_method import *
# from sklearn.decomposition import PCA
# #
# #
# def pca(X, Y, k):  # k is the components you want
#     # mean of each feature
#     n_samples, n_features = X.shape
#     mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
#     # normalization
#     norm_X = X - mean
#     mean2 = np.array([np.mean(Y[:, i]) for i in range(n_features)])
#     norm_Y = Y - mean2
#     # scatter matrix
#     scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
#     scatter_matrix_Y = np.dot(np.transpose(norm_Y), norm_Y)
#     pca = PCA(n_components=2)
#     pca.fit(X)
#     pca2 = PCA(n_components=2)
#     pca2.fit(Y)
#
#     # Calculate the eigenvectors and eigenvalues
#     eig_val, eig_vec = np.linalg.eig(scatter_matrix)
#     eig_val2, eig_vec2 = np.linalg.eig(scatter_matrix_Y)
#     print("scatter_matrix:")
#     print(scatter_matrix)
#     print("eig_val:")
#     print(eig_val)
#     print("eig_vec:")
#     print(eig_vec)
#     print("scatter_matrix_Y:")
#     print(scatter_matrix_Y)
#     print("eig_val_Y:")
#     print(eig_val2)
#     print("eig_vec_Y:")
#     print(eig_vec2)
#     eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
#     # sort eig_vec based on eig_val from highest to lowest
#     eig_pairs.sort(reverse=True)
#     eig_pairs2 = [(np.abs(eig_val2[i]), eig_vec2[:, i]) for i in range(n_features)]
#     # sort eig_vec based on eig_val from highest to lowest
#     eig_pairs2.sort(reverse=True)
#     # select the top k eig_vec
#     feature = np.array([ele[1] for ele in eig_pairs[:k]])
#     feature2 = np.array([ele[1] for ele in eig_pairs2[:k]])
#     print("feature:")
#     print(feature)
#     # get new data
#     print("feature[0]:")
#     print(feature[0])
#     data = np.dot(scatter_matrix, np.transpose(feature))
#     data2 = np.dot(scatter_matrix_Y, np.transpose(feature))
#     data3 = np.dot(scatter_matrix_Y, np.transpose(feature2))
#     print("val:")
#     print(data)
#     import math
#     a2 = data[0][0] * data[0][0] + data[0][1] * data[0][1]
#     a2 = math.sqrt(a2)
#     a3 = data3[0][0] * data3[0][0] + data3[0][1] * data3[0][1]
#     a3 = math.sqrt(a3)
#     print("a2:")
#     print(a2)
#     print("a3:")
#     print(a3)
#     # print()
#     a = pca.transform(scatter_matrix)
#     b = pca2.transform(scatter_matrix_Y)
#     print(a)
#     print(b)
#     print(math.sqrt(a[0][0] * a[0][0] + a[0][1] * a[0][1]))
#     print(math.sqrt(b[0][0] * b[0][0] + b[0][1] * b[0][1]))
#     # print(np.dot(norm_X, np.transpose(feature)))
#     # print(pca.transform(X))
#     return data
#
#
# import time
#
# a = np.zeros((120, 160))
# b = np.zeros((120, 160))
# for i in range(120):
#     for j in range(160):
#         a[i][j] = random.random()
#         b[i][j] = random.random()
#
# # X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# # Y = np.array([[-6, 3], [0, -1], [0, -2], [1, 0], [2, 0], [3, 0]])
# s = time.time()
# # pca(X, Y, 2)
# print(cal_single(a, b))
# e = time.time()
# print(e - s)
# # s = time.time()
# # pca(X, Y, 2)
# # print(cal_single(a, b))
# # print(cal_single_by_sklearn(a,b))
# # e = time.time()
# # print(e - s)
# #
# # '''
# # 3.3
# # Label5:FedAvg*10
# #
# # TODO
# # Label5:entropy:bound/alpha
# #     a=0.1 b=0.2
# #     a=5 b=0.2
# #     a=5 b=10
# #
# # TODO:
# # Label5:cos*10
# # Label5:l2*10
# # Label5:entropy:bound/alpha
# # '''
# # d = {"1": [1], "2": [2]}
# # print(d)
# # d["1"] = [4]
# # print(d)
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import math
# #
# #
# # class SimpleCNN(nn.Module):
# #     def __init__(self, input_dim, hidden_dims, output_dim=10):
# #         super(SimpleCNN, self).__init__()
# #         self.conv1 = nn.Conv2d(3, 6, 5)
# #         self.pool = nn.MaxPool2d(2, 2)
# #         self.conv2 = nn.Conv2d(6, 16, 5)
# #
# #         # for now, we hard coded this network
# #         # i.e. we fix the number of hidden layers i.e. 2 layers
# #         self.fc1 = nn.Linear(input_dim, hidden_dims[0])
# #         self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
# #         self.fc3 = nn.Linear(hidden_dims[1], output_dim)
# #
# #     def forward(self, x):
# #         x = self.pool(F.relu(self.conv1(x)))
# #         x = self.pool(F.relu(self.conv2(x)))
# #         x = x.view(-1, 16 * 5 * 5)
# #
# #         x = F.relu(self.fc1(x))
# #         x = F.relu(self.fc2(x))
# #         x = self.fc3(x)
# #         return x
# #
# #
# # net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
# # p = net.state_dict()
# # for param in p.keys():
# #     print(param)
# import torch
# from torchvision.transforms import transforms, functional
#
# pattern_tensor = torch.tensor([
#     [1., 0., 1.],
#     [-10., 1., -10.],
#     [-10., -10., 0.],
#     [-10., 1., -10.],
#     [1., 0., 1.]
# ])
#
# x_top = 3
# "X coordinate to put the backdoor into."
# y_top = 23
# "Y coordinate to put the backdoor into."
#
# mask_value = -10
# "A tensor coordinate with this value won't be applied to the image."
# input_shape = (32, 32, 3)
# full_image = torch.zeros((32, 32, 3))
# full_image.fill_(mask_value)
# print(len(input_shape))
# x_bot = x_top + pattern_tensor.shape[0]
# y_bot = y_top + pattern_tensor.shape[1]
#
#
# if x_bot >= input_shape[0] or \
#         y_bot >= input_shape[1]:
#     raise ValueError(f'Position of backdoor outside image limits:'
#                      f'image: {input_shape}, but backdoor'
#                      f'ends at ({x_bot}, {y_bot})')
#
# full_image[x_top:x_bot, y_top:y_bot,:] = pattern_tensor
#
# # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                                  std=[0.229, 0.224, 0.225])
# normalize = transforms.Normalize((0.1307,), (0.3081,))
# # full_image = full_image.unsqueeze_(0)
import csv #调用数据保存文件
import pandas as pd #用于数据输出
y1_j= [12.697433731817252, 29.20946279943478, 49.67323524323002, 102.88872277028304, 172.95152378588483, 260.14917353058166]
y2_j= [12.697434705656148, 29.209335842559813, 49.67162354953921, 102.8596739269148, 172.8617729038298, 260.0592364430679]
y3_j= [12.697444267517733, 29.208966696508188, 49.66134738870873, 102.71200463740381, 172.47114709328892, 259.72051120522076]
y4_j= [12.697523854073596, 29.2076660189428, 49.630150039544326, 102.33955478334542, 171.58165579048637, 258.97900442099444]
y5_j= [12.697391090950354, 29.196235034586824, 49.412658430997, 100.12739415822013, 166.30174161735337, 247.94405307296105]
y6_j= [12.701194603318868, 29.149956734930655, 48.655819938382166, 91.77462349516475, 139.48350457473015, 189.6119072414784]
y7_j= [13.901056630066016, 29.006123485565933, 46.542109857216175, 72.78566488101947, 83.5985050921565, 80.75520583184539]
list_jmax=[y1_j,y2_j,y3_j,y4_j,y5_j,y6_j,y7_j]
column=[r for r in range(6)] #列表头名称
test=pd.DataFrame(columns=column,data=list_jmax)
test.to_csv('./123445test.csv') #存储位置及文件名称
import numpy as np

import copy
import os
import gc
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.data import *
from src.models import *
from src.fedavg import *
from src.client import *
from src.clustering import *
from src.utils import *
from src.entropy_method import *

args = args_parser()

args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

torch.cuda.set_device(args.gpu)  ## Setting cuda on GPU


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


path = args.savedir + args.alg + '/' + args.partition + '/' + args.dataset + '/'
mkdirs(path)

template = "Algorithm {}, Clients {}, Dataset {}, Model {}, Non-IID {}, Threshold {}, K {}, Linkage {}, LR {}, Ep {}, Rounds {}, bs {}, frac {},Alpha {}"

s = template.format(args.alg, args.num_users, args.dataset, args.model, args.partition, args.cluster_alpha,
                    args.n_basis, args.linkage, args.lr, args.local_ep, args.rounds, args.local_ep, args.frac,
                    args.alpha)

print(s)

print(str(args))
results = []
for times in range(args.times):
    ##################################### Data partitioning section
    args.local_view = True
    X_train, y_train, X_test, y_test, net_dataidx_map, net_dataidx_map_test, \
    traindata_cls_counts, testdata_cls_counts = partition_data(args.dataset,
                                                               args.datadir, args.logdir, args.partition,
                                                               args.num_users,
                                                               beta=args.beta, local_view=args.local_view)

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                      args.datadir,
                                                                                      args.batch_size,
                                                                                      32)
    examples = enumerate(train_dl_global)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)

    print("len train_ds_global:", len(train_ds_global))
    print("len test_ds_global:", len(test_ds_global))


    ################################### build model
    def init_nets(args, dropout_p=0.5):
        users_model = []

        for net_i in range(-1, args.num_users):
            if args.dataset == "generated":
                net = PerceptronModel().to(args.device)
            elif args.model == "mlp":
                if args.dataset == 'covtype':
                    input_size = 54
                    output_size = 2
                    hidden_sizes = [32, 16, 8]
                elif args.dataset == 'a9a':
                    input_size = 123
                    output_size = 2
                    hidden_sizes = [32, 16, 8]
                elif args.dataset == 'rcv1':
                    input_size = 47236
                    output_size = 2
                    hidden_sizes = [32, 16, 8]
                elif args.dataset == 'SUSY':
                    input_size = 18
                    output_size = 2
                    hidden_sizes = [16, 8]
                net = FcNet(input_size, hidden_sizes, output_size, dropout_p).to(args.device)
            elif args.model == "vgg":
                net = vgg11().to(args.device)
            elif args.model == "simple-cnn":
                if args.dataset in ("cifar10", "cinic10", "svhn"):
                    net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(args.device)
                elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                    net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).to(args.device)
                elif args.dataset == 'celeba':
                    net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2).to(args.device)
            elif args.model == "simple-fc":
                if args.dataset in ("mnist", 'femnist', 'fmnist'):
                    net = SimpleCNNMNIST(input_dim=(28 * 28), hidden_dims=[300, 100], output_dim=10).to(args.device)
            elif args.model == "simple-cnn-3":
                if args.dataset == 'cifar100':
                    net = SimpleCNN_3(input_dim=(16 * 3 * 5 * 5), hidden_dims=[120 * 3, 84 * 3], output_dim=100).to(
                        args.device)
                if args.dataset == 'tinyimagenet':
                    net = SimpleCNNTinyImagenet_3(input_dim=(16 * 3 * 13 * 13), hidden_dims=[120 * 3, 84 * 3],
                                                  output_dim=200).to(args.device)
            elif args.model == "vgg-9":
                if args.dataset in ("mnist", 'femnist'):
                    net = ModerateCNNMNIST().to(args.device)
                elif args.dataset in ("cifar10", "cinic10", "svhn"):
                    # print("in moderate cnn")
                    net = ModerateCNN().to(args.device)
                elif args.dataset == 'celeba':
                    net = ModerateCNN(output_dim=2).to(args.device)
            elif args.model == 'resnet9':
                if args.dataset == 'cifar100':
                    net = ResNet9(in_channels=3, num_classes=100)
                elif args.dataset == 'tinyimagenet':
                    net = ResNet9(in_channels=3, num_classes=200, dim=512 * 2 * 2)
            elif args.model == "resnet":
                net = ResNet50_cifar10().to(args.device)
            elif args.model == "vgg16":
                net = vgg16().to(args.device)
            else:
                print("not supported yet")
                exit(1)
            if net_i == -1:
                net_glob = copy.deepcopy(net)
                initial_state_dict = copy.deepcopy(net_glob.state_dict())
                server_state_dict = copy.deepcopy(net_glob.state_dict())
                if args.load_initial:
                    initial_state_dict = torch.load(args.load_initial)
                    server_state_dict = torch.load(args.load_initial)
                    net_glob.load_state_dict(initial_state_dict)
            else:
                users_model.append(copy.deepcopy(net))
                users_model[net_i].load_state_dict(initial_state_dict)

        return users_model, net_glob, initial_state_dict, server_state_dict


    print(f'MODEL: {args.model}, Dataset: {args.dataset}')

    users_model, net_glob, initial_state_dict, server_state_dict = init_nets(args, dropout_p=0.5)

    print(net_glob)

    total = 0
    for name, param in net_glob.named_parameters():
        print(name, param.size())
        total += np.prod(param.size())
    print(total)

    ################################# Initializing Clients
    clients = []

    K = args.n_basis

    for idx in range(args.num_users):
        # net dataidx map里面放的是数据的下标
        dataidxs = net_dataidx_map[idx]
        if net_dataidx_map_test is None:
            dataidx_test = None
        else:
            dataidxs_test = net_dataidx_map_test[idx]

        # print(f'Initializing Client {idx}')

        noise_level = args.noise
        if idx == args.num_users - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset,
                                                                                          args.datadir, args.local_bs,
                                                                                          32,
                                                                                          dataidxs, noise_level, idx,
                                                                                          args.num_users - 1,
                                                                                          dataidxs_test=dataidxs_test)
        else:
            noise_level = args.noise / (args.num_users - 1) * idx
            train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset,
                                                                                          args.datadir, args.local_bs,
                                                                                          32,
                                                                                          dataidxs, noise_level,
                                                                                          dataidxs_test=dataidxs_test)

        if idx < args.flipnum:
            for index in range(len(train_ds_local.target)):
                img = train_ds_local.data[index]
                img = img / 255
                mean = 0
                sigma = args.noise2
                # 产生高斯 noise
                noise = np.random.normal(mean, sigma, img.shape)
                # 将噪声和图片叠加
                gaussian_out = img + noise
                # 将超过 1 的置 1，低于 0 的置 0
                gaussian_out = np.clip(gaussian_out, 0, 1)
                # 将图片灰度范围的恢复为 0-255
                gaussian_out = np.uint8(gaussian_out * 255)
                train_ds_local.data[index] = torch.from_numpy(gaussian_out)

        train_dl_local = data.DataLoader(dataset=train_ds_local, batch_size=args.local_bs, shuffle=True,
                                         drop_last=False)
        test_dl_local = data.DataLoader(dataset=test_ds_local, batch_size=args.local_bs, shuffle=False, drop_last=False)

        clients.append(Client_FedAvg(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep,
                                     args.lr, args.momentum, args.device, train_dl_local, test_dl_local))

    ###################################### Federation

    loss_train = []

    init_tracc_pr = []  # initial train accuracy for each round
    final_tracc_pr = []  # final train accuracy for each round

    init_tacc_pr = []  # initial test accuarcy for each round
    final_tacc_pr = []  # final test accuracy for each round

    init_tloss_pr = []  # initial test loss for each round
    final_tloss_pr = []  # final test loss for each round

    clients_best_acc = [0 for _ in range(args.num_users)]
    w_locals, loss_locals = [], []

    init_local_tacc = []  # initial local test accuracy at each round
    final_local_tacc = []  # final local test accuracy at each round

    init_local_tloss = []  # initial local test loss at each round
    final_local_tloss = []  # final local test loss at each round

    ckp_avg_tacc = []
    ckp_avg_best_tacc = []

    users_best_acc = [0 for _ in range(args.num_users)]
    best_glob_acc = 0

    w_glob = copy.deepcopy(initial_state_dict)
    print_flag = False
    for iteration in range(args.rounds):

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # idxs_users = comm_users[iteration]

        print(f'###### ROUND {iteration + 1} ######')
        print(f'Clients {idxs_users}')

        for idx in idxs_users:
            clients[idx].set_state_dict(copy.deepcopy(w_glob))

            loss = clients[idx].train(is_print=False)

            loss_locals.append(copy.deepcopy(loss))

        total_data_points = sum([len(net_dataidx_map[r]) for r in idxs_users])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in idxs_users]

        w_locals = []
        w_locals2 = []
        for idx in idxs_users:
            w_locals.append(copy.deepcopy(clients[idx].get_state_dict()))

        ww = FedAvg(w_locals, weight_avg=fed_avg_freqs)
        w_glob = copy.deepcopy(ww)
        net_glob.load_state_dict(copy.deepcopy(ww))
        _, acc = eval_test(net_glob, args, test_dl_global)
        if acc > best_glob_acc:
            best_glob_acc = acc

            # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)

        print('## END OF ROUND ##')
        template = 'Average Train loss {:.3f}'
        print(template.format(loss_avg))

        template = "Global Model Test Loss: {:.3f}, Global Model Test Acc: {:.3f}, Global Model Best Test Acc: {:.3f}"
        print(template.format(_, acc, best_glob_acc))

        print_flag = False
        if iteration % args.print_freq == 0:
            print_flag = True

        if print_flag:
            print('--- PRINTING ALL CLIENTS STATUS ---')
            current_acc = []
            for k in range(args.num_users):
                loss, acc = clients[k].eval_test()
                current_acc.append(acc)

                if acc > clients_best_acc[k]:
                    clients_best_acc[k] = acc

                template = ("Client {:3d}, labels {}, count {}, best_acc {:3.3f}, current_acc {:3.3f} \n")
                print(template.format(k, traindata_cls_counts[k], clients[k].get_count(),
                                      clients_best_acc[k], current_acc[-1]))

            template = ("Round {:1d}, Avg current_acc {:3.3f}, Avg best_acc {:3.3f}")
            print(template.format(iteration + 1, np.mean(current_acc), np.mean(clients_best_acc)))

            ckp_avg_tacc.append(np.mean(current_acc))
            ckp_avg_best_tacc.append(np.mean(clients_best_acc))

        print('----- Analysis End of Round -------')
        for idx in idxs_users:
            print(f'Client {idx}, Count: {clients[idx].get_count()}, Labels: {traindata_cls_counts[idx]}')

        loss_train.append(loss_avg)
        loss_locals.clear()
        init_local_tacc.clear()
        init_local_tloss.clear()
        final_local_tacc.clear()
        final_local_tloss.clear()

        ## calling garbage collector
        gc.collect()

    test_loss = []
    test_acc = []
    train_loss = []
    train_acc = []

    for idx in range(args.num_users):
        loss, acc = clients[idx].eval_test()

        test_loss.append(loss)
        test_acc.append(acc)

        loss, acc = clients[idx].eval_train()

        train_loss.append(loss)
        train_acc.append(acc)

    test_loss = sum(test_loss) / len(test_loss)
    test_acc = sum(test_acc) / len(test_acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_acc) / len(train_acc)

    print(f'Train Loss: {train_loss}, Test_loss: {test_loss}')
    print(f'Train Acc: {train_acc}, Test Acc: {test_acc}')

    print(f'Best Clients AVG Acc: {np.mean(clients_best_acc)}')

    net_glob.load_state_dict(copy.deepcopy(w_glob))
    _, acc = eval_test(net_glob, args, test_dl_global)
    if acc > best_glob_acc:
        best_glob_acc = acc

    template = "Global Model Test Acc: {:.3f}, Global Model Best Test Acc: {:.3f}"
    print(template.format(acc, best_glob_acc))
    results.append(best_glob_acc)
    ############################# Saving Print Results
    with open(path + str(args.trial) + '_final_results.txt', 'a') as text_file:
        print(f'Train Loss: {train_loss}, Test_loss: {test_loss}', file=text_file)
        print(f'Train Acc: {train_acc}, Test Acc: {test_acc}', file=text_file)

        print(f'Best Clients AVG Acc: {np.mean(clients_best_acc)}', file=text_file)

        template = "Global Model Test Acc: {:.3f}, Global Model Best Test Acc: {:.3f}"
        print(template.format(acc, best_glob_acc), file=text_file)

    avg_result = np.mean(results)
    var_result = np.var(results)
    with open(path + str(args.trial) + '_beta=' + str(args.beta) + '_flipnum=' + str(args.flipnum) + '_results_record.txt',
              'w') as text_file:
        print(results, file=text_file)
        print(avg_result, file=text_file)
        print(var_result, file=text_file)

import warnings

warnings.filterwarnings('ignore')

import torch
from torchvision.models import resnet18
from torchvision.models import vgg16

import copy
import time
import numpy as np
import random
from tqdm import trange
from scipy import signal

from utils.options import args_parser
from utils.sampling import cifar_iid
from utils.sampling import noniid
from utils.sampling import gpt_noniid
from utils.dataset import load_data
from utils.dataset import change_label
from utils.test import test_img
from src.aggregation import server_opt
from src.update import EdgeOpt
from src.models import LeNet5
# 打开文件，清空内容


def extract_parameters_to_vector(model):

    params = []
    for param in model.parameters():
        # 将参数移动到 GPU 上
        params.append(param.data.cuda().view(-1))
    # 连接成一个向量
    param_vector = torch.cat(params, 0)
    return param_vector


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset_train, dataset_test = load_data(args)

    once_classes = 2
    times = 100 // once_classes

    args.num_classes = once_classes
    data_dict = {}
    pre_net_glob = resnet18(num_classes=args.num_classes).to(args.device)
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        pre_net_glob.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    ################################################################################################################## pre_train

    args.num_pretrain = 1           # 之前的预训练是为了选matrix of B, 所以每次的模型都是新的。现在为了保留模型，只预训练一次就行
    for num_pre in range(args.num_pretrain):


        '''
        pre_net_glob = LeNet5().to(args.device)
        if args.dataset == 'cifar':
            pre_net_glob.conv1 = torch.nn.Conv2d(3, 6, 5)
            pre_net_glob.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        '''

        num_params = sum(p.numel() for p in pre_net_glob.parameters())
        pre_param_vector = extract_parameters_to_vector(pre_net_glob)
        pre_last_net = pre_net_glob


        total_params = sum(p.numel() for p in pre_net_glob.parameters())

        #b_vector = np.concatenate([np.random.rand(np.prod(shape)) for shape in param_shapes])
        #pre_B = np.array(10, np.concatenate([np.random.rand(np.prod(shape)) for shape in param_shapes]))
        original_pre_B = np.random.rand(10, total_params)

        for i in range(4):  # 类增
            pre_net_glob.train()
            if i == 0:
                pre_B = np.zeros_like(original_pre_B)
            else:
                pre_B = original_pre_B
            train_classes = []
            for j in range(once_classes):
                train_classes.append(i * once_classes + j)

            if i == 1:
                train_classes = [29, 30]
                test_classes = [29, 30]
            elif i == 2:
                train_classes = [7, 8]
                test_classes = [7, 8]
            elif i == 3:
                train_classes = [19, 20]
                test_classes = [19, 20]



            # filter dataset
            filtered_dataset_train = [(image, label) for image, label in dataset_train if label in train_classes]

            # 修改训练集中的标签
            filtered_dataset_train = change_label(filtered_dataset_train, train_classes)

            dict_users = cifar_iid(filtered_dataset_train, args.num_clients)  # 100 users
            #dict_users = gpt_noniid(filtered_dataset_train, args.num_clients, args.sampling_classes, args.num_classes)

            # 为了fedrs模拟源代码中的rand_class_num
            o_classes = np.random.randint(0, args.num_classes, size=(args.num_clients, args.sampling_classes))

            # cyclic server learning rate
            the_time = np.arange(0, 2, 0.01)
            freq = args.freq
            max_clr = 0
            clr_cycle = args.amp * signal.sawtooth(2 * np.pi * freq * the_time)
            clr = 1.0 - clr_cycle

            pre_w_glob = pre_net_glob.state_dict()

            for iter in trange(args.global_ep):
                w_locals = []
                selected_clients = max(int(args.frac * args.num_clients), 1)
                idxs_users = np.random.choice(range(args.num_clients), selected_clients, replace=False)

                b_idx = 0   #记录第几个用户，idx是从100个里面随机选的
                for idx in idxs_users:
                    local = EdgeOpt(args=args, dataset=filtered_dataset_train, idxs=dict_users[idx],user_classes=o_classes[idx])
                    #
                    w = local.train(global_net=copy.deepcopy(pre_net_glob).to(args.device),net=copy.deepcopy(pre_net_glob).to(args.device), b_vector=pre_B[b_idx], last_round_vector=pre_param_vector.to(args.device), rounds=i,last_net= copy.deepcopy(pre_last_net).to(args.device))
                    #w = local.train_sgd(previous_net=None, global_net=copy.deepcopy(pre_net_glob).to(args.device),net=copy.deepcopy(pre_net_glob).to(args.device),b_vector=pre_B[b_idx],last_round_vector=pre_param_vector,rounds=i)
                    w_locals.append(copy.deepcopy(w))
                    
                    if iter == args.global_ep-1:


                        local_net = resnet18(num_classes=args.num_classes).to(args.device)
                        if args.dataset == 'mnist' or args.dataset == 'fmnist':
                            local_net.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                             bias=False)
                                                             
                        '''

                        local_net= LeNet5().to(args.device)
                        if args.dataset == 'cifar':
                            local_net.conv1 = torch.nn.Conv2d(3, 6, 5)
                            local_net.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
                        '''


                        local_net.load_state_dict(w)
                        local_net.eval()                # local_net 不参与训练，训练用的copy.deepcopy(pre_net_glob)，所以不用改为train，一直eval就行
                        test_classes = train_classes
                        filtered_dataset_test = [(image, label) for image, label in dataset_test if label in test_classes]
                        filtered_dataset_test = change_label(filtered_dataset_test, test_classes)
                        test_acc, test_loss = test_img(local_net.to(args.device), filtered_dataset_test, args)
                        #print("test loss is ",test_loss)
                        if b_idx == 0:
                            best_loss_user = test_loss
                            best_b_users = pre_B[b_idx]
                            #print("best loss is ", best_loss_user)
                        else:
                            if test_loss < best_loss_user:
                                best_loss_user = test_loss
                                best_b_users = pre_B[b_idx]
                                #print("best loss is ", best_loss_user)
                    b_idx = b_idx + 1

                # update global weights
                pre_w_glob = server_opt(w_locals, args)

                # copy weight to net_glob
                pre_net_glob.load_state_dict(pre_w_glob)

            if i == 1:
                avg_loss = best_loss_user
                avg_b = best_b_users
                #print("average loss is ", avg_loss)
            elif i>1:
                avg_loss = (avg_loss * (i-1) + best_loss_user)/i
                avg_b = (avg_b * (i-1) + best_b_users)/i


            # 训练完更新一下
            pre_param_vector = extract_parameters_to_vector(pre_net_glob)
            pre_last_net = pre_net_glob

        '''
        filename = f'./matrix_b/array_data_{num_pre}.npy'
        np.save(filename, avg_b)
        
        with open('loss.txt', 'a') as file:
            file.write(str(avg_loss) + "\n")
        '''

        print("pre_train round ", num_pre ," is finished, ", "average loss is ", avg_loss)
        if num_pre == 0:
            best_b = avg_b
            best_loss = avg_loss
            print("b in first time has built!!!!!!!!!!!!!!!!!!the best_loss is ", best_loss)
            np.save('best_b.npy', best_b)
        else:
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_b = avg_b
                print("b has updated!!!!!!!!!!!!!!, the best loss is ",best_loss)
                np.save('best_b.npy', best_b)

    ##########################################################################################################################  pre_train has finished



    '''

    net_glob = LeNet5().to(args.device)
    if args.dataset == 'cifar':
        net_glob.conv1 = torch.nn.Conv2d(3, 6, 5)
        net_glob.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
    '''

    #param_vector = extract_parameters_to_vector(net_glob)
    #last_net = net_glob



    for i in range(4):
        # build model

        net_glob = resnet18(num_classes=args.num_classes).to(args.device)       # 用空白的模型
        #net_glob = pre_net_glob        # 用预训练的模型
        if args.dataset == 'mnist' or args.dataset == 'fmnist':
            net_glob.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)



        net_glob.train()

        train_classes = []
        test_classes = []
        for j in range(once_classes):
            train_classes.append( i * once_classes + j)
            test_classes.append( i * once_classes + j)

        if i == 1:
            train_classes = [29, 30]
            test_classes = [29, 30]
        elif i == 2:
            train_classes = [7, 8]
            test_classes = [7, 8]
        elif i == 3:
            train_classes = [19, 20]
            test_classes = [19, 20]
        #print(f"train_classes is {train_classes}")
        #print(f"test_classes is {test_classes}")


        # filter dataset
        filtered_dataset_train = [(image, label) for image, label in dataset_train if label in train_classes]

        # 修改训练集中的标签
        filtered_dataset_train = change_label(filtered_dataset_train, train_classes)

        dict_users = cifar_iid(filtered_dataset_train, args.num_clients)  # 100 users
        #dict_users = gpt_noniid(filtered_dataset_train, args.num_clients, args.sampling_classes, args.num_classes)

        # 为了fedrs模拟源代码中的rand_class_num
        o_classes = np.random.randint(0, args.num_classes, size=(args.num_clients, args.sampling_classes))

        # cyclic server learning rate
        the_time = np.arange(0, 2, 0.01)
        freq = args.freq
        max_clr = 0
        clr_cycle = args.amp * signal.sawtooth(2 * np.pi * freq * the_time)
        clr = 1.0 - clr_cycle

        # copy weights
        w_glob = net_glob.state_dict()

        for iter in trange(1000):

            net_glob.train()
            w_locals = []
            selected_clients = max(int(args.frac * args.num_clients), 1)
            idxs_users = np.random.choice(range(args.num_clients), selected_clients, replace=False)

            b_idx = 0
            for idx in idxs_users:
                local = EdgeOpt(args=args, dataset=filtered_dataset_train, idxs=dict_users[idx], user_classes=o_classes[idx])
                w = local.train(global_net=copy.deepcopy(net_glob).to(args.device),net=copy.deepcopy(net_glob).to(args.device), b_vector=best_b, rounds=i,last_net = copy.deepcopy(last_net).to(args.device))
                #w = local.old_train(previous_net=None, global_net=copy.deepcopy(net_glob).to(args.device),net=copy.deepcopy(net_glob).to(args.device))
                b_idx = b_idx + 1
                w_locals.append(copy.deepcopy(w))

            # update global weights
            w_glob = server_opt(w_locals, args)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # test
            net_glob.eval()
            filtered_dataset_test = [(image, label) for image, label in dataset_test if label in test_classes]
            filtered_dataset_test = change_label(filtered_dataset_test, test_classes)
            test_acc, test_loss = test_img(net_glob.to(args.device), filtered_dataset_test, args)
            if test_acc > 85:
                print(f"Test accuracy in {test_classes} is {test_acc}, the iter is {iter}")
                break



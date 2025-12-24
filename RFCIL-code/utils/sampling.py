import numpy as np
from torch.utils.data import Subset
from utils.options import args_parser


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    args = args_parser()

    #num_items = int(len(dataset)/num_users)
    num_items = args.num_data
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def noniid(dataset, args):
    idxs = np.arange(len(dataset))
    
    
    if args.dataset == 'mnist':
        labels = dataset.targets.numpy()
    elif args.dataset == 'fmnist':
        labels = dataset.targets.numpy()
    elif args.dataset == 'cifar':
        labels = np.array(dataset.targets)
    elif args.dataset == 'svhn':
        labels = np.array(dataset.labels)
    else:
        exit('Error: unrecognized dataset')

    dict_users = {i: list() for i in range(args.num_clients)}
    dict_labels = dict()

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    
    idxs = list(idxs_labels[0])
    labels = idxs_labels[1]
    
    rand_class_num = np.random.randint(0, args.num_classes, size= (args.num_clients, args.sampling_classes))
    print(rand_class_num)

    for i in range(args.num_classes):
        specific_class = set(np.extract(labels == i, idxs))
        dict_labels.update({i : specific_class})

    for i, class_num in enumerate(rand_class_num):
        rand_set = list()
        
        rand_class1 = list(np.random.choice(list(dict_labels[class_num[0]]), args.num_data))
        rand_class2 = list(np.random.choice(list(dict_labels[class_num[1]]), args.num_data))
        #rand_class3 = list(np.random.choice(list(dict_labels[class_num[2]]), args.num_data))
        #rand_class4 = list(np.random.choice(list(dict_labels[class_num[3]]), args.num_data))
        
        dict_labels[class_num[0]] = dict_labels[class_num[0]] - set(rand_class1)
        dict_labels[class_num[1]] = dict_labels[class_num[1]] - set(rand_class2)
        #dict_labels[class_num[2]] = dict_labels[class_num[2]] - set(rand_class3)
        #dict_labels[class_num[3]] = dict_labels[class_num[3]] - set(rand_class4)
 
        rand_set = rand_set + rand_class1 + rand_class2 #+ rand_class3 + rand_class4
        dict_users[i] = set(rand_set)

    return dict_users, rand_class_num





def gpt_noniid(dataset, num_users, sampling_classes, num_classes):
    """
    Split dataset among users to simulate non-IID data distribution.
    This function distributes the data so that each user gets data
    from only a random subset of the total classes, with overlap between users.

    Args:
    - dataset: list of (image, label) tuples representing the dataset.
    - num_users: number of users to distribute the data among.
    - sampling_classes: number of classes to sample for each user.

    Returns:
    - dict_users: dictionary mapping user index to list of data indices they own.
    - o_classes: numpy array representing the classes each user owns.
    """
    num_samples_per_user = len(dataset) // num_users

    dict_users = {i: [] for i in range(num_users)}
    o_classes = np.zeros((num_users, sampling_classes), dtype=int)

    for user_id in range(num_users):
        classes_for_user = np.random.choice(range(num_classes), sampling_classes, replace=False)
        o_classes[user_id] = classes_for_user

        for class_id in classes_for_user:
            class_indices = [i for i in range(len(dataset)) if
                             dataset[i][1] == class_id]
            selected_indices = np.random.choice(class_indices, num_samples_per_user // sampling_classes, replace=False)
            dict_users[user_id].extend(selected_indices)

    # Adjust for any rounding issues to ensure all users get a fair share of data
    remaining_indices = set(range(len(dataset))) - set([i for sublist in dict_users.values() for i in sublist])
    remaining_indices = list(remaining_indices)
    np.random.shuffle(remaining_indices)
    while remaining_indices:
        for user_id in range(num_users):
            if not remaining_indices:
                break
            dict_users[user_id].append(remaining_indices.pop())

    return dict_users
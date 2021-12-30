import torch
import torchvision
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler
from PIL import Image
import json

import json
import os, copy
import random
from torch.utils.data import DataLoader, Dataset
import numpy as np


class MNISTDataset(Dataset):
    """EMNIST dataset"""
    def __init__(self, feature, target, transform=None):
        # self.X = []
        self.Y = target
        self.transform = transform
        self.X = feature
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.X[idx]), self.Y[idx]
        return self.X[idx], self.Y[idx]


def femnist_dataloaders(root="./femnist", batch_size=64, clients=10):
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_data = torch.load(os.path.join(root, "train_data.pt"))
    test_data = torch.load(os.path.join(root, "test_data.pt"))
    
    # flatten image format
    if isinstance(clients, int):
        print("Filter clients by position.")
        if clients > len(train_data['users']):
            raise ValueError(
                "Request clients({}) larger than dataset provide({}).".format(clients, len(train_data['users'])))

        train_data['users'] = train_data['users'][:clients]
        test_data['users'] = test_data['users'][:clients]
    elif isinstance(clients, list):
        print("Filter clients by name_list.")
        for name in clients:
            if name not in train_data['users']:
                raise ValueError("Client {} not found in dataset.".format(name))
        train_data['users'] = clients
    
    ##############
    # Preprocess #
    ##############
    train_data_all = {'x': [], 'y': []}
    train_loaders = []
    
    test_data_all = {'x': [], 'y': []}
    test_loaders = []
    
    # cut data to fit batch size and using dataloader to batch data by its batch size
    for user in train_data['users']:
        # calculate x, y cut size
        x_cut_size = int(len(train_data['user_data'][user]['x']) / batch_size) * batch_size
        y_cut_size = int(len(train_data['user_data'][user]['y']) / batch_size) * batch_size
        
        # get user feature x, target y
        train_x = train_data['user_data'][user]['x'][:x_cut_size]
        train_y = train_data['user_data'][user]['y'][:y_cut_size]
        test_x = test_data['user_data'][user]['x'][:x_cut_size]
        test_y = test_data['user_data'][user]['y'][:y_cut_size]
        
        # save all user data 
        train_data_all['x'] += train_x
        train_data_all['y'] += train_y
        test_data_all['x'] += test_x
        test_data_all['y'] += test_y
        
        # using dataset and dataloader to set data into batch
        train_dataset = MNISTDataset(torch.tensor(train_x).view(-1, 28, 28), torch.tensor(train_y), data_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
        
        test_dataset = MNISTDataset(torch.tensor(test_x).view(-1, 28, 28), torch.tensor(test_y), data_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        test_loaders.append(test_loader)
    
    # data_global set
    train_dataset_global = MNISTDataset(torch.tensor(train_data_all['x']).view(-1, 28, 28), torch.tensor(train_data_all['y']), data_transform)
    train_data_global = DataLoader(train_dataset_global, batch_size=batch_size, shuffle=True)
    
    test_dataset_global = MNISTDataset(torch.tensor(test_data_all['x']).view(-1, 28, 28), torch.tensor(test_data_all['y']), data_transform)
    test_data_global = DataLoader(test_dataset_global, batch_size=batch_size, shuffle=True)
    
    # data_local_dict set
    train_data_local_dict = {}
    for i,v in enumerate(train_loaders):
        train_data_local_dict[i] = v
        
    test_data_local_dict = {}
    for i,v in enumerate(test_loaders):
        test_data_local_dict[i] = v    
    
    # data num set
    train_data_num = 0
    test_data_num = 0
    train_data_local_num_dict = {}
    
    for k in train_data_local_dict.keys():
        train_data_local_num_dict[k] = len(train_data_local_dict[k].dataset)
        train_data_num += len(train_loaders[k].dataset)
    
    for k in test_data_local_dict.keys():
        test_data_num += len(test_loaders[k].dataset)
    
    # dict format:
    
    # train_data = {
    #   "users": ['f4015_05', 'f4067_23', ......],
    #   "num_samples": [154, 162, 121, ......]
    #   "user_data": {
    #       'f4015_05':{
    #           "x": [[1.0, 0.99, 0.59, 1.0, 0.96, 0.41, ,...], [...]],
    #           "y": [1,12,4, ....],
    #                  }
    #       'f4067_23':{
    #           "x": [[0.98, 0.99, 0.19, 1.0, 0.76, 0.42, ,...], [...]],
    #           "y": [19,22,48, ....],
    #                  }
    #  }
    #}
    
    # total 3560 clients.
    # In LAB3 we take first 50 clients
    
    #############################################################################
    
    # example dataloader
    # example_dataset = MNISTDataset(feature=torch.tensor(train_data['user_data']['f4015_05']['x']).view(-1, 28, 28), target = torch.tensor(train_data['user_data']['f4015_05']['y']), transform=data_transform)
    # example_dataloader = DataLoader(example_dataset, batch_size=32, shuffle=True)

    # [
    #     train_data_num,             -> total image in train dataset
    #     test_data_num,              -> total image in test dataset
    #     train_data_global,          -> dataloader with all train dataset
    #     test_data_global,           -> dataloader with all test dataset
    #     train_data_local_num_dict,  -> dict of amount of train data in each clients. ex: {0:123, 1:80, 2:231, ...}
    #     train_data_local_dict,      -> dict of amount of train data in each clients. ex: {0:<dataloader>, 1:<dataloader>, ...}
    #     test_data_local_dict,       -> dict of amount of test data in each clients. ex: {0:<dataloader>, 1:<dataloader>, ...}
    #     class_num                   -> number of class. femnist: 62
    # ]
    
    
    dataset =  [
        train_data_num, 
        test_data_num, 
        train_data_global, 
        test_data_global,
        train_data_local_num_dict, 
        train_data_local_dict, 
        test_data_local_dict, 
        62
    ]
    return dataset
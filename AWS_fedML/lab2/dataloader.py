import json
import os, copy
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import SubsetRandomSampler
import torch.multiprocessing
from PIL import Image
import numpy as np

class SHDataset(Dataset):
    def __init__(self, feature, target, transform=None):
        self.X = torch.tensor(feature)
        self.Y = torch.tensor(target)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def shakespeare_dataloaders(root="./femnist", batch_size=4, clients=20):
    train_data = torch.load(os.path.join(root, "train_data.pt"))
    test_data = torch.load(os.path.join(root, "test_data.pt"))
    # ['users', 'num_samples', 'user_data', 'hierarchies']
    if clients > len(train_data['users']):
        raise ValueError(
            "Request clients({}) larger than dataset provide({}).".format(clients, len(train_data['users'])))
    train_data['users'] = train_data['users'][:clients]
    test_data['users'] = test_data['users'][:clients]
    #############################################################################
    train_data_all_x = []
    train_data_all_y = []
    train_idx = []

    trainloaders = []
    for u in train_data["users"]:
        train_data["user_data"][u]['x']
        x = [word_to_indices(sen) for sen in train_data["user_data"][u]['x']]
        x = x[:int(len(x)/batch_size)*batch_size]

        y = [word_to_indices(sen)[0] for sen in train_data["user_data"][u]['y']]
        y = y[:int(len(y) / batch_size) * batch_size]
        train_data_all_x += x
        train_data_all_y += y

        traloader = torch.utils.data.DataLoader(SHDataset(x, y), batch_size=batch_size, shuffle=False)
        trainloaders.append(traloader)

    test_data_all_x = []
    test_data_all_y = []
    test_idx = []
    testloaders = []
    for i in test_data["users"]:
        # cut data to fit batch
        x = [word_to_indices(sen) for sen in test_data["user_data"][i]["x"]]
        x = x[:int(len(x) / batch_size) * batch_size]

        y = [word_to_indices(sen)[0] for sen in test_data["user_data"][i]["y"]]
        y = y[:int(len(y) / batch_size) * batch_size]
        test_data_all_x += x
        test_data_all_y += y
        test_idx.append(len(x))
        
        testloader = torch.utils.data.DataLoader(SHDataset(x, y), batch_size=batch_size, shuffle=False)
        testloaders.append(testloader)
    #############################################################################
    train_dataset = SHDataset(train_data_all_x, train_data_all_y)
    train_data_global = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = SHDataset(test_data_all_x, test_data_all_y)
    test_data_global = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    #############################################################################
    
    train_data_local_dict = {}
    for i,v in enumerate(trainloaders):
        train_data_local_dict[i] = v

    
    test_data_local_dict = {}
    for i,v in enumerate(testloaders):
        test_data_local_dict[i] = v        
    #############################################################################
    train_data_local_num_dict = {}
    
    train_data_num = 0
    for k in train_data_local_dict.keys():
        train_data_local_num_dict[k] = len(train_data_local_dict[k].dataset)
        train_data_num += len(trainloaders[k].dataset)

    test_data_num = len(testloader.dataset)
    #############################################################################
    

    #return {"test": testloader, "train_s": trainloaders}
    dataset = [train_data_num, 
               test_data_num, 
               train_data_global,
               test_data_global,
               train_data_local_num_dict,
               train_data_local_dict,
               test_data_local_dict,
               80
              ]
    return dataset
# [
#     train_data_num, 
#     test_data_num, 
#     train_data_global, 
#     test_data_global,
#     train_data_local_num_dict, 
#     train_data_local_dict, 
#     test_data_local_dict, 
#     class_num
# ]
def word_to_indices(word):
    ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices

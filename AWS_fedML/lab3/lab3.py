import argparse
import logging
import os
import random
import sys
import time
import json
from typing_extensions import Required
import numpy as np
import copy


import torch
import torchvision
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler
import wandb

from dataloader import femnist_dataloaders


# from fedml_api.model.cv.resnet import resnet56
from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI
from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer
from model import Net_femnist

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--wandb_name', type=str, required=True,
                        help='Name of log file')
    
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./cifar10',
                        help='data directory')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
                        
    parser.add_argument('--seed', type=int, default=0,
                        help='seed')
    return parser


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    # parser.add_argument("-f")

    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    wandb.init(
        project="fedml",
        name=args.wandb_name,
        config=args
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    dataset = femnist_dataloaders(root="./femnist",clients=args.client_num_in_total, batch_size=args.batch_size)
    

    model = Net_femnist()

    dummy_opt = torch.optim.SGD(copy.deepcopy(model).parameters(), lr=args.lr)
    dummy_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=dummy_opt, 
                                                    step_size = args.comm_round/5,
                                                    gamma = 0.5)
    model_trainer = MyModelTrainer(model)
    fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer, threading=3, scheduler=dummy_scheduler)
    #fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer, scheduler=dummy_scheduler)
    fedavgAPI.train()

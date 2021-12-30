import copy
import glob
import json
import os
import sys
import time

import torch


if __name__ == '__main__':

    if os.path.exists(os.path.abspath("./femnist")):
        raise ValueError("Folder: \"{}\" exist".format(os.path.abspath("./femnist")))
    os.makedirs(os.path.abspath("./femnist"))
    import gdown, tarfile
    # download
    url = 'https://drive.google.com/uc?id=1tzA5b92qhrFBjVTyVP645Dfs4fTV2SNL'
    output = os.path.join(os.path.abspath("./femnist"), 'femnist_100up_jsonfile.tar.gz')
    print("\nDownload ...")
    gdown.download(url, output, quiet=False)
    # check
    md5 = 'df9e06065616a86228ceea8ab690012d'
    gdown.cached_download(url, output, md5=md5, postprocess=gdown.extractall)
    time.sleep(3)
    # extraction
    print("\nExtracting ...")
    tar = tarfile.open(output, 'r:gz')
    tar.extractall(path=os.path.abspath("./femnist"))

    path = os.path.abspath("./femnist/femnist_100up_jsonfile")
    
    train_list = glob.glob(os.path.join(path, "train", "all_data_*.json"))
    test_list = glob.glob(os.path.join(path, "test", "all_data_*.json"))

    train_list = sorted(train_list, key = lambda train_list : int(train_list.split("/")[-1].split("_")[2]))
    test_list = sorted(test_list, key = lambda test_list : int(test_list.split("/")[-1].split("_")[2]))
    
    # for isl FedML tutorial, only selecte first 300 clients
    train_list = train_list[:1]
    test_list = test_list[:1]

    train_data = {'users': [], 'num_samples': [], 'user_data': {}}
    test_data = {'users': [], 'num_samples': [], 'user_data': {}}

    for f in train_list:
        print("Load : {}".format(f))
        file_ = open(f, 'r')
        context = json.load(file_)
        file_.close()
        train_data['users'] += context['users']
        train_data['num_samples'] += context['num_samples']
        train_data['user_data'].update(context['user_data'])

    for f in test_list:
        print("Load : {}".format(f))
        file_ = open(f, 'r')
        context = json.load(file_)
        file_.close()
        test_data['users'] += context['users']
        test_data['num_samples'] += context['num_samples']
        test_data['user_data'].update(context['user_data'])

    os.makedirs(os.path.join(os.path.abspath("."), "femnist"), exist_ok=True)
    torch.save(train_data, os.path.join(os.path.abspath("."), "femnist", "train_data.pt"))
    print("\nSave : {}".format(os.path.join(os.path.abspath("."), "femnist", "train_data.pt")))

    torch.save(test_data, os.path.join(os.path.abspath("."), "femnist", "test_data.pt"))
    print("\nSave : {}".format(os.path.join(os.path.abspath("."), "femnist", "test_data.pt")))

    train_data_name = {'users': copy.deepcopy(test_data['users']), }
    for name in train_data["users"]:
        for i, _ in enumerate(train_data['user_data'][name]['x']):
            train_data['user_data'][name]['x'][i] = [0]
    torch.save(train_data, os.path.join(os.path.abspath("."), "femnist", "train_data_name_target_only.pt"))
    print("\nSave : {}".format(os.path.join(os.path.abspath("."), "femnist", "train_data_name_target_only.pt")))

    time.sleep(3)
    print("Done")

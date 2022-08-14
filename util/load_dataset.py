import natsort
import glob
import json
import collections
import os

import torch
import random
import numpy as np

seed = 32
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(seed)

def jsonfileparser(filepath, encoding='utf-8'):
    print('parse information from json file')
    with open(filepath, encoding=encoding) as file:
        dict = json.load(file, object_pairs_hook=collections.OrderedDict)

    return dict


def GetAllFileList(file_path, ext):
    all_files = natsort.natsorted((glob.glob(os.path.join(file_path, ext))))
    return all_files




class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, x_train, t_train):
        self.x_train = x_train
        self.t_train = t_train

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return torch.tensor(self.x_train[idx], dtype=torch.float32), \
               torch.tensor(int(self.t_train[idx]), dtype=torch.int64)

class ValidDataset(torch.utils.data.Dataset):

    def __init__(self, x_valid, t_valid):
        self.x_valid = x_valid
        self.t_valid = t_valid

    def __len__(self):
        return len(self.x_valid)

    def __getitem__(self, idx):
        return torch.tensor(self.x_valid[idx], dtype=torch.float32), \
               torch.tensor(int(self.t_valid[idx]), dtype=torch.int64)


# データをtorch.tensorに変換する
class TestDataset(torch.utils.data.Dataset):

    def __init__(self, x_test, t_test):
        self.x_test = x_test
        self.t_test = t_test

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, idx):
        return torch.tensor(self.x_test[idx], dtype=torch.float32), \
               torch.tensor(int(self.t_test[idx]), dtype=torch.int64)



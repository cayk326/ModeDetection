import collections
import os

import seaborn as sns; sns.set()
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from util.preprocessing import generate_batch, batch_norm
from util.load_dataset import GetAllFileList, jsonfileparser, TrainDataset, TestDataset, ValidDataset
from models import LSTM_Classification
from engine.train import train_model
import numpy as np
import random
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1' #  There are always errors below  shape  atypism 

seed = 32
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
#os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)
Isdocker = True

class Settings:
    def __init__(self):
        if Isdocker:
            self.confpath = "/workspaces/ModeDetection/config/config_docker.json"
        else:
            self.confpath = "D:/PythonCode/ModeDetection/config/config.json"
        self.config = None
        self.all_files_path = None

def main():
    print("Start Main")
    settings = Settings()

    settings.config = jsonfileparser(settings.confpath)
    print("Creating dataset file list")
    x_train_dataset_list = GetAllFileList(settings.config["System"]["TrainInputFileDir"] + '/x_train', '*.csv')
    y_train_dataset_list = GetAllFileList(settings.config["System"]["TrainInputFileDir"] + '/y_train', '*.csv')

    x_valid_dataset_list = GetAllFileList(settings.config["System"]["ValidInputFileDir"] + '/x_valid', '*.csv')
    y_valid_dataset_list = GetAllFileList(settings.config["System"]["ValidInputFileDir"] + '/y_valid', '*.csv')

    x_test_dataset_list = GetAllFileList(settings.config["System"]["TestInputFileDir"] + '/x_test', '*.csv')
    y_test_dataset_list = GetAllFileList(settings.config["System"]["TestInputFileDir"] + '/y_test', '*.csv')

    print('-------------------Dataset Information----------------------')
    print("Number of x_train dataset {0}".format(len(x_train_dataset_list)))
    print("Number of y_train dataset {0}".format(len(y_train_dataset_list)))

    print("Number of x_valid dataset {0}".format(len(x_valid_dataset_list)))
    print("Number of y_valid dataset {0}".format(len(y_valid_dataset_list)))

    print("Number of x_test dataset {0}".format(len(x_test_dataset_list)))
    print("Number of y_test dataset {0}".format(len(y_test_dataset_list)))
    print('-------------------------------------------------------------')

    print('-----------------create batch for train----------------------')
    x_train_batch, y_train_batch = generate_batch(x_train_dataset_list, y_train_dataset_list, settings)
    print('-----------------fit and transform x train batch for standardization----------------------')
    bn = batch_norm()
    # z_score normalization(mean=0, std,sigma=1)
    # データの最大値、最小値は不明でこれらの値にスケーリング後のデータが引っ張られないようにする
    #x_train_batch, x_train_mu, x_train_std = bn.fit_transform(x_train_batch)

    print('-------------------create train loader-----------------------')
    train_dataset = TrainDataset(x_train_batch, y_train_batch)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=settings.config["ModelParams"]["batch_size"], shuffle=True, worker_init_fn=seed_worker, generator=g)


    print('-----------------create batch for validation----------------------')
    x_val_batch, y_val_batch = generate_batch(x_valid_dataset_list, y_valid_dataset_list, settings)
    print('-----------------transform x val batch for standardization using scale of train----------------------')
    #x_val_batch= bn.transform(x_val_batch, x_train_mu, x_train_std)

    print('-------------------create val loader-----------------------')
    val_dataset = ValidDataset(x_val_batch, y_val_batch)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=settings.config["ModelParams"]["batch_size"], shuffle=False)


    print('------------------Model Configuration-----------------------')
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Use device: {0}'.format(DEVICE))
    LSTM_Model = LSTM_Classification.LSTM(seq_len=x_train_batch.shape[1],
                                          feature_size=x_train_batch.shape[2],
                                          hidden_dim=settings.config["ModelParams"]["hidden_dim"],
                                          num_lstm_layers=settings.config["ModelParams"]["num_lstm_layer"],
                                          out_dim=settings.config["ModelParams"]["out_dim"],
                                          dropout_ratio=settings.config["ModelParams"]["dropout_ratio"],
                                          classification=settings.config["ModelParams"]["classification"]
                                          )
    LSTM_Model.to(DEVICE)
    optimizer = optim.AdamW(LSTM_Model.parameters(), lr=settings.config["ModelParams"]["lr"])
    loss_fn = nn.CrossEntropyLoss()

    LSTM_Model.train()
    LSTM_Model.to(DEVICE)

    import transformers.optimization as Trans_optim
    LRFinderFlag=settings.config['ModelParams']['LR_Finder_Flag']
    scheduler = None
    if eval(LRFinderFlag):
        # スケジューラ設定
        total_steps = len(x_train_batch) * settings.config['ModelParams']['epochs']

        scheduler = Trans_optim.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
    elif eval(settings.config['ModelParams']['scheduler_flag']):
        # Learning rate decay scheduler
        # scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20, eta_min=0.0001)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=settings.config["ModelParams"]["scheduler_gamma"])
        # スケジューラ設定
        WARM_UP_RATIO = 0.1
        total_steps = len(x_train_batch) * settings.config['ModelParams']['epochs']
        n_warmup = int(total_steps * WARM_UP_RATIO)
        scheduler = Trans_optim.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=n_warmup,
            num_training_steps=total_steps
        )

    Isvisdom = eval(settings.config['System']['Visdom'])
    if Isvisdom:
        import visdom
        logger = visdom.Visdom()
        print()
    else:
        logger = []
    print('------------------All preparation was Completed-----------------------')


    print('Start training...')
    IsNeedTrain = True
    IsNeedTest = True
    if IsNeedTrain:
        try:
            model, res_train_loss, res_train_acc, res_val_loss, res_val_acc, res_train_f1, res_val_f1 = train_model(settings, LSTM_Model, optimizer, loss_fn, settings.config["ModelParams"]["epochs"], scheduler, train_loader, val_loader, logger)
        except Exception as e:
            #torch.save(model.state_dict(), settings.config["System"]["OutputFileDir"] + '\\models\\LSTM_Model_Error.pt')
            #torch.jit.script(model).save(settings.config["System"]["OutputFileDir"] + '\\models\\Jitted_LSTM_Model_Error.pt')
            print(e)

    if IsNeedTest:
        print('Final evaluation phase for generalization of model...')
        print('-----------------create batch for test----------------------')
        x_test_batch, y_test_batch = generate_batch(x_test_dataset_list, y_test_dataset_list, settings)

        print('-----------------transform x test batch for standardization using scale of train----------------------')
        #x_test_batch = bn.transform(x_test_batch, x_train_mu, x_train_std)

        print('-------------------create val loader-----------------------')
        test_dataset = TestDataset(x_test_batch, y_test_batch)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=settings.config["ModelParams"]["batch_size"], shuffle=False)
        from engine import test
        print('Load Jitted Model...')
        tuned_model = torch.jit.load(settings.config["System"]["OutputFileDir"] + '/models/Jitted_LSTM_Model.pt')
        print('Load Jitted Model completed!')
        tuned_model.to(DEVICE)

        test.testing(settings, tuned_model, test_loader, loss_fn, logger)


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    end = time.time()
    print(end - start)
from ast import IsNot
import collections
from distutils.debug import DEBUG
import multiprocessing
import seaborn as sns; sns.set()
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import subprocess
from util.preprocessing import generate_batch, batch_norm
from util.load_dataset import GetAllFileList, jsonfileparser, TrainDataset, TestDataset, ValidDataset
from models import LSTM_Classification
from engine.train import train_model
import numpy as np
import transformers.optimization as Trans_optim
from engine.make_graph import traingraph
import  random
import os
seed = 32
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(seed)



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

def optmain():
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
    #bn = batch_norm()
    #x_train_batch, x_train_mu, x_train_std = bn.fit_transform(x_train_batch)

    print('-------------------create train loader-----------------------')
    train_dataset = TrainDataset(x_train_batch, y_train_batch)

    print('-----------------create batch for validation----------------------')
    x_val_batch, y_val_batch = generate_batch(x_valid_dataset_list, y_valid_dataset_list, settings)
    print('-----------------transform x val batch for standardization using scale of train----------------------')
    #x_val_batch= bn.transform(x_val_batch, x_train_mu, x_train_std)

    print('-------------------create val loader-----------------------')
    val_dataset = ValidDataset(x_val_batch, y_val_batch)

    seq_len = x_train_batch.shape[1]
    feature_size = x_train_batch.shape[2]
    out_dim = settings.config["ModelParams"]["out_dim"]
    hyper_param_optimization(train_dataset, val_dataset, settings, seq_len, feature_size, out_dim)




import optuna


# objective
def hyper_param_optimization(train_dataset, val_dataset, settings, seq_len, feature_size, out_dim):
    def objective(trial):

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("DEVICE is {0}".format(DEVICE))


        BATCH_SIZE = trial.suggest_int("BATCH_SIZE", settings.config["HP_Optim"]["start_batch_size"],
                                              settings.config["HP_Optim"]["end_batch_size"])

        NUM_EPOCHS = trial.suggest_int("NUM_EPOCHS", settings.config["HP_Optim"]["start_epochs"],
                                               settings.config["HP_Optim"]["end_epochs"])
        HIDDEN_DIM = trial.suggest_int("HIDDEN_DIM", settings.config["HP_Optim"]["start_hidden_dim"],
                                               settings.config["HP_Optim"]["end_hidden_dim"])

        LSTM_LAYER = trial.suggest_int("LSTM_LAYER", settings.config["HP_Optim"]["start_num_lstm_layer"],
                                               settings.config["HP_Optim"]["end_num_lstm_layer"])

        LR = trial.suggest_loguniform("LR", settings.config["HP_Optim"]["start_lr"],
                                               settings.config["HP_Optim"]["end_lr"])

        #DROPOUT_RATIO = trial.suggest_loguniform("DROPOUT_RATIO", settings.config["HP_Optim"]["start_dropout_ratio"],
        #                  settings.config["HP_Optim"]["end_dropout_ratio"])

        DROPOUT_RATIO=0

        #WEIGHT_DECAY = trial.suggest_loguniform("WEIGHT_DECAY", settings.config["HP_Optim"]["start_weight_decay"],
        #                               settings.config["HP_Optim"]["end_weight_decay"])

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   worker_init_fn=seed_worker,
                                                   generator=g)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=BATCH_SIZE, shuffle=False)

        logger = []
        scheduler = None

        print("HIDDEN_DIM is: " + str(HIDDEN_DIM))
        print("LSTM_LAYER is: " + str(LSTM_LAYER))
        print("BATCH_SIZE is: " + str(BATCH_SIZE))
        print("NUM_EPOCHS is: " + str(NUM_EPOCHS))
        print("LR is: " + str(LR))
        print("DROPOUT_RATIO is: " + str(DROPOUT_RATIO))
        #print("WEIGHT_DECAY is " + str(WEIGHT_DECAY))

        LSTM_Model = LSTM_Classification.LSTM(seq_len=seq_len,
                                              feature_size=feature_size,
                                              hidden_dim=HIDDEN_DIM,
                                              num_lstm_layers=LSTM_LAYER,
                                              out_dim=out_dim,
                                              dropout_ratio=DROPOUT_RATIO,
                                              classification=settings.config["ModelParams"]["classification"]
                                              )



        LSTM_Model.to(DEVICE)
        optimizer = optim.AdamW(LSTM_Model.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()


        LRFinderFlag = settings.config['ModelParams']['LR_Finder_Flag']
        if eval(LRFinderFlag):
            # スケジューラ設定
            total_steps = train_loader.dataset.x_train.shape[0] * NUM_EPOCHS

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
            total_steps = train_loader.dataset.x_train.shape[0] * NUM_EPOCHS
            n_warmup = int(total_steps * WARM_UP_RATIO)
            scheduler = Trans_optim.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=n_warmup,
                num_training_steps=total_steps
            )


        LSTM_Model.train()
        LSTM_Model.to(DEVICE)

        model, res_train_loss, res_train_acc, res_val_loss, res_val_acc, res_train_f1, res_val_f1 = train_model(settings, LSTM_Model, optimizer, loss_fn, NUM_EPOCHS, scheduler, train_loader, val_loader, logger)
        """
        ハイパーパラメータチューニング結果の情報を出力
        trial._trial_id
        """

        #torch.save(model.state_dict(), settings.config["System"]["OutputFileDir"] + '\\models\\'  +'trialID_' + str(trial._trial_id) + '_LSTM_Model.pt')
        torch.jit.script(model).save(
            settings.config["System"]["OutputFileDir"] + '/models/' + 'trialID_' + str(trial._trial_id) + '_LSTM_Model.pt')

        return (-1) * res_val_f1[-1]#res_val_loss[-1]

    """
    TPE base Bayesian optimization

    """
    study_name = "optimization"
    storage = "sqlite:///optimization.db"

    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=100)
    #sampler = optuna.samplers.TPESampler()# TPE method
    sampler = optuna.samplers.CmaEsSampler()  # CmaEs method
    timeout = 60 * 60 * settings.config["HP_Optim"]["optim_time_hour"]
    # timeout=600
    # n_trials = 50
    # sampler = optuna.samplers.CmaEsSampler()

    subprocess.Popen(["optuna-dashboard", storage])
    study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage, load_if_exists=True,
                                sampler=sampler, pruner=pruner)
    # study.optimize(objective, n_trials=50)
    study.optimize(objective, timeout=timeout)

    print(study.best_value)
    print(study.best_params)
    return study


if __name__ == '__main__':
    import datetime
    import threading
    DEBUGMode = False
    THREAD_NUM = 5

    print(str(datetime.datetime.now()).replace("-","").replace(" ","_").replace(":","").replace(".","_"))
    thread_list = []
    if DEBUGMode == False:
        for i in range(THREAD_NUM):
            thread_list.append(threading.Thread(target=optmain))
        for i in range(THREAD_NUM):
            thread_list[i].start()
        
        print('Optimization was finished.')
    else:
        print('Debug mode')
        optmain()


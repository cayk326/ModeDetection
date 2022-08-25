'''
This function for actual predicting task using trained model
'''

import collections
import seaborn as sns;

sns.set()
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from engine.make_graph import predgraph

from util.preprocessing import generate_batch
from util.load_dataset import GetAllFileList, jsonfileparser, TrainDataset, TestDataset, ValidDataset
from models import LSTM_Classification
from engine.train import train_model
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score

# import visdom
import statistics

Isdocker = True
class Settings:
    def __init__(self):
        if Isdocker:
            self.confpath = "/workspaces/ModeDetection/config/config_docker.json"
        else:
            self.confpath = "D:/PythonCode/ModeDetection/config/config.json"
        self.config = None
        self.all_files_path = None


def load_model(model_path, model):
    print('load model: ' + str(model_path))
    model.load_state_dict(torch.load(model_path))
    print('Finished to load')
    return model


'''
This is for test function to evaluate how robust developed model
'''
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score


def prediction(settings, model, test_loader, loss_fn):
    print('Start testing...')
    model.eval()

    test_loss = 0
    num_classes = settings.config["ModelParams"]["num_classes"]
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    confusion = np.zeros((num_classes, num_classes))
    pred_label = []
    pred_label_list = []  # 各イタレーションのラベル推論結果を格納
    true_label_list = []

    for iteration, (x, t) in enumerate(test_loader):
        x, t = x.to(DEVICE), t.to(DEVICE)
        y = model(x)
        test_loss += loss_fn(y, t)  # 各イタレーションの検証誤差を累積する

        rows = t.cpu().numpy()
        cols = y.argmax(1).cpu().numpy()
        pred_label_list = np.hstack((pred_label_list, cols))  # 推論ラベルを格納
        true_label_list = np.hstack((true_label_list, rows))  # 正解ラベルを格納

    confusion = confusion_matrix(true_label_list, pred_label_list, labels=[i for i in range(num_classes)])  # テスト混同行列の算出
    avg_test_loss = test_loss / iteration  # 累積された検証誤差をイタレーションで割り平均を出す。検証誤差の最終値

    acc = accuracy_score(true_label_list, pred_label_list)  # テストフェーズにおける推論精度
    recall = recall_score(true_label_list, pred_label_list, average="macro")  # Recall
    precision = precision_score(true_label_list, pred_label_list, average="macro")  # Precision
    f1 = f1_score(true_label_list, pred_label_list, average="macro")  # f1

    print("★Final Result :average_test_loss : {loss} | test_acc : {acc}, "
          "test_precision: {precision}, test_recall: {recall}, test_f1: {f1} ".format(
        loss=avg_test_loss,
        acc=acc,
        precision=precision,
        recall=recall,
        f1=f1
    )
    )
    print("Confusion Matrix for a data")
    print(confusion)
    try:
        final_pred_label = statistics.mode(pred_label_list)
    except Exception as e:
        import collections
        print('model returned multiple value as prediction')
        final_pred_label = collections.Counter(pred_label_list).most_common()[0][0]
    '''
    # Plot confusion matrix in visdom
    # ABCDE = 01234
    logger.heatmap(confusion, win='10', opts=dict(
        title="Test_Confusion_Matrix",
        columnnames=["A", "B", "C", "D", "E"],
        rownames=["A", "B", "C", "D", "E"])
                   )

    '''
    return avg_test_loss, acc, final_pred_label


def pred_main():
    print("Start Main module")
    settings = Settings()
    settings.config = jsonfileparser(settings.confpath)

    print("Creating test dataset file list")
    x_test_dataset_list = GetAllFileList(settings.config["System"]["PredictionInputFileDir"] + '/x_test', '*.csv')
    y_test_dataset_list = GetAllFileList(settings.config["System"]["PredictionInputFileDir"] + '/y_test', '*.csv')

    print('-------------------Dataset Information----------------------')
    print("Number of x_test dataset {0}".format(len(x_test_dataset_list)))
    print("Number of y_test dataset {0}".format(len(y_test_dataset_list)))
    print('-------------------------------------------------------------')

    num_classes = settings.config["ModelParams"]["num_classes"]
    # 実際の分類における性能を評価するためのマトリクス
    # 訓練、検証、テストで実施するような導出と異なり、1ファイルを分析した結果の最終出力値のみを見る
    Final_confusion = np.zeros((num_classes, num_classes))

    final_pred_label_list = []
    final_true_label_list = []

    for x_test_dataset, y_test_dataset in zip(x_test_dataset_list, y_test_dataset_list):
        print(x_test_dataset)
        print('-----------------create batch for test----------------------')
        x_test_batch, y_test_batch = generate_batch([x_test_dataset], [y_test_dataset], settings)

        print('-------------------create val loader-----------------------')
        test_dataset = TestDataset(x_test_batch, y_test_batch)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=settings.config["ModelParams"]["batch_size"],
                                                  shuffle=False)

        print('Prediction phase...')

        print('------------------Model Configuration-----------------------')
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tuned_model = LSTM_Classification.LSTM(seq_len=x_test_batch.shape[1],
                                               feature_size=x_test_batch.shape[2],
                                               hidden_dim=settings.config["ModelParams"]["hidden_dim"],
                                               num_lstm_layers=settings.config["ModelParams"]["num_lstm_layer"],
                                               out_dim=settings.config["ModelParams"]["out_dim"],
                                               dropout_ratio=0,
                                               classification=settings.config["ModelParams"]["classification"])

        load_model(settings.config["System"]["OutputFileDir"] + '/models/LSTM_Model.pt', tuned_model)
        loss_fn = nn.CrossEntropyLoss()

        tuned_model.to(DEVICE)
        avg_test_loss, acc, final_pred_label = prediction(settings, tuned_model, test_loader, loss_fn)
        Final_confusion[int(y_test_batch[0]), int(final_pred_label)] += 1  # rows行 cols列に1を加算。rowsは正解値、colsは予測値
        final_pred_label_list.append(int(final_pred_label))
        final_true_label_list.append(int(y_test_batch[0]))

        if final_pred_label == 0:
            print('You are AA Rank...pred label is {0}'.format(final_pred_label))
        elif final_pred_label == 1:
            print('You are BB Rank...pred label is {0}'.format(final_pred_label))
        elif final_pred_label == 2:
            print('You are CC Rank...pred label is {0}'.format(final_pred_label))
        elif final_pred_label == 3:
            print('You are DD Rank...pred label is {0}'.format(final_pred_label))
        else:
            print('You are EE Rank...pred label is {0}'.format(final_pred_label))


    print("-----------------")
    print("Final Confusion Matrix")
    print(Final_confusion)
    acc = np.trace(Final_confusion) / np.sum(Final_confusion)
    f1 = f1_score(final_true_label_list, final_pred_label_list, average="macro")
    predgraph(f1, Final_confusion, settings)


if __name__ == '__main__':
    pred_main()

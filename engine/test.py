'''
This is for test function to evaluate how robust developed model
'''
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from .make_graph import testgraph
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
from util.tools import get_confusion_matrix


def testing(settings, model, test_loader, loss_fn, logger):
    print('Start testing...')
    Isvisdom = eval(settings.config['System']['Visdom'])
    model.eval()

    test_loss = 0
    num_classes = settings.config["ModelParams"]["num_classes"]
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    confusion = np.zeros((num_classes, num_classes))
    pred_label_list = [] # 全テストデータに対する推論結果を格納したリスト
    true_label_list = [] # 全テストデータに対する正解ラベルを格納したリスト
    for iteration, (x, t) in enumerate(test_loader):
        x, t = x.to(DEVICE), t.to(DEVICE)
        y = model(x)
        test_loss += loss_fn(y, t)  # 各イタレーションのテストデータに対する推論誤差を累積する

        rows = t.cpu().numpy() # 各イタレーションにおけるテストデータの正解ラベル
        cols = y.argmax(1).cpu().numpy()# 各イタレーションにおけるテストデータに対する推論結果

        pred_label_list = np.hstack((pred_label_list, cols))  # 推論ラベルを格納
        true_label_list = np.hstack((true_label_list, rows))  # 正解ラベルを格納

    confusion = get_confusion_matrix(true_label_list, pred_label_list, settings.config['ModelParams']['out_dim'])  # テストデータに対する混同行列の算出

    avg_test_loss = test_loss / iteration #累積されたテスト誤差をイタレーションで割り平均を出す。テストデータに対する誤差の最終値

    acc = accuracy_score(true_label_list, pred_label_list)# テストフェーズにおける推論精度
    recall = recall_score(true_label_list, pred_label_list, average="macro")  # Recall
    precision = precision_score(true_label_list, pred_label_list, average="macro")  # Precision
    f1 = f1_score(true_label_list, pred_label_list, average="macro")  # f1

    print("★Final Result :average_test_loss : {loss} | test_acc : {acc}, "
        "test_precision: {precision}, test_recall: {recall}, test_f1: {f1} ".format(
        loss = avg_test_loss,
        acc = acc,
        precision = precision,
        recall = recall,
        f1 = f1
        )
    )

    testgraph(avg_test_loss, f1, confusion, settings)

    return avg_test_loss, acc


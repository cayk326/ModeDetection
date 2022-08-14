import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .inference import validation
import numpy as np
from .make_graph import traingraph
from ModeDetection.util.tools import get_confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
import json
import datetime

import random
import os

seed = 32
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(seed)
def train_model(settings, model, optimizer, loss_fn, num_epoch, scheduler, train_loader, val_loader, logger):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    res_train_loss = []#平均訓練ロスを全て配列にして保存
    res_val_loss = []#平均検証ロスを全て配列にして保存
    res_train_acc = []# 平均訓練制度
    res_val_acc = []# 平均検証精度

    res_train_recall = []# Recall
    res_train_precision = []# Precision
    res_train_f1 = []# f1

    res_val_precision = []
    res_val_recall = []
    res_val_f1 = []


    y_train_pred_list = []# 各イタレーションの推測結果のリスト
    t_train_list = []# 各イタレーションの正解値のリスト

    lr_list = []# 学習率リスト
    Isvisdom = eval(settings.config['System']['Visdom'])
    num_classes = settings.config["ModelParams"]["num_classes"]

    x_axis_epoch = [i for i in range(1, num_epoch + 1)]
    val_acc_best = 0
    for epoch in range(1, num_epoch + 1):
        iter_train_list = []# 各エポックにおける全イタレーションのロス
        train_epoch_loss = 0  # 各エポックにおける累積訓練ロス
        train_confusion = np.zeros((num_classes, num_classes))  # 訓練用混同行列

        pred_label_list = []  # 各イタレーションのラベル推論結果を格納
        true_label_list = []  # 各イタレーションの正解ラベルを格納

        for iteration, (x, t) in enumerate(train_loader):
            x, t = x.to(DEVICE), t.to(DEVICE)
            optimizer.zero_grad()  # 勾配の初期化

            y = model(x)  # 予測の計算(順伝播)
            train_loss = loss_fn(y, t)  # 損失関数の計算
            train_loss.backward()  # 勾配の計算（逆伝播）

            # 勾配クリッピング
            nn.utils.clip_grad_norm_(model.parameters(), settings.config['ModelParams']['max_norm'])

            optimizer.step()  # 重みの更新
            if eval(settings.config['ModelParams']['scheduler_flag']):
                scheduler.step() # スケジューラの更新
                lr_list.append(scheduler.get_last_lr()[0])# 更新した学習率を格納
            else:
                lr_list.append(settings.config["ModelParams"]["lr"])

            iter_train_list.append(train_loss.item())# 角イタレーションごとの訓練ロスを格納
            train_epoch_loss += train_loss.item()# 訓練ロスの合計。訓練終了後に平均訓練ロスを算出


            pred = y.argmax(1)  # n項分類においてモデルが推定した、各カテゴリである確率のベクトルをもとに最大値のインデックスを抽出する


            rows_t = t.cpu().numpy()# True value
            cols_pred = pred.cpu().numpy()# Pred value



            """
            pred_label_listとtrue_label_listは学習セット全体、すなわち1エポックで消費する学習データの長さだけ
            推測ラベルと正解ラベルを管理するリスト
            cols_pred, rows_tはある1エポックにおける1エタレーションの推論結果と正解ラベル
            """
            pred_label_list = np.hstack((pred_label_list, cols_pred))
            true_label_list = np.hstack((true_label_list, rows_t))


            if iteration % 10 == 0:
                print("Epoch: {ep}/{max}, iter: {iter}, loss: {loss}, LR:{lr}".format(
                    ep=epoch,
                    max=num_epoch,
                    iter=iteration,
                    loss=train_loss,
                    lr = lr_list[-1])
                )
            del train_loss

        """
        1エポックごとの混同行列を更新する
        この時、1エポックにおける
        """
        train_confusion = get_confusion_matrix(truelabel=true_label_list, predlabel=pred_label_list,
                                               num_classes=settings.config['ModelParams']['out_dim'])
        #train_confusion = confusion_matrix(true_label_list, pred_label_list)# 訓練混同行列の算出
        avg_train_loss = train_epoch_loss / iteration#各エポックの平均訓練ロス
        res_train_loss.append(avg_train_loss)

        #train_acc = np.trace(train_confusion) / np.sum(train_confusion)  # 1エポック当たりの訓練データに対する推論精度
        """
        評価指標
        macro平均を用いる。ただし、macro平均はデータのクラスに偏りがあると正しく評価できない
        そのため、入力するデータにはクラスの偏りが内容にする必要がある
        データのクラスの偏りが少なければaccuracyモデルの傾向を把握することができる
        """

        res_train_acc.append(accuracy_score(true_label_list, pred_label_list))# 各エポックの推論精度を格納
        res_train_recall.append(recall_score(true_label_list, pred_label_list, average="macro"))  # Recall
        res_train_precision.append(precision_score(true_label_list, pred_label_list, average="macro"))# Precision
        res_train_f1.append(f1_score(true_label_list, pred_label_list, average="macro"))# F1 Score


        print("★Epoch: {ep}/{max}, average_train_loss: {loss}, train_acc: {train_acc}, "
              "train_precision: {train_precision}, train_recall: {train_recall}, train_f1: {train_f1} ".format(
            ep=epoch,
            max=num_epoch,
            loss=avg_train_loss,
            train_acc=res_train_acc[-1],
            train_precision=res_train_precision[-1],
            train_recall=res_train_recall[-1],
            train_f1=res_train_f1[-1]
            )
        )


        # バッチ処理に対応するためインデックスを-1に指定
        # 各バッチの最終予測値が採用される
        """
        各Epochのイタレーションの最終実行結果における推測ラベルと正解ラベルを管理する
        y_train_pred_listで学習が推移するにおいて、推論ラベルが正解ラベルにあってくるかわかる
        あまり使わない
        """
        y_train_pred_list.append(int(cols_pred[-1]))# 各エポックの最終予測値 = 列
        t_train_list.append(int(rows_t[-1]))# 各エポックの最終正解値 = 行


        # 各エポックごとに訓練データで学習後のモデルに対してに検証データを使って性能検証する
        val_loss, val_acc, val_precision, val_recall, val_f1, val_confusion = validation(settings, model, val_loader, loss_fn, epoch, num_epoch, logger)
        res_val_loss.append(val_loss)
        res_val_acc.append(val_acc)
        res_val_precision.append(val_precision)
        res_val_recall.append(val_recall)
        res_val_f1.append(val_f1)


        if val_acc_best < val_acc:
            val_acc_best = val_acc
            #torch.save(model.state_dict(), settings.config["System"]["OutputFileDir"] + '\\models\\' + str(epoch) +'_LSTM_Model.pt')
            torch.jit.script(model).save(settings.config["System"]["OutputFileDir"] + '\\models\\' + 'Gen' + str(epoch) + '_Jitted_LSTM_Model.pt')

        # 訓練モードに戻す
        model.train()

        if Isvisdom:
            logger.line([avg_train_loss],
                             [epoch],
                             opts=dict(title="Training_loss"),
                             win='1',
                             name='train_loss',
                             update='append'
                    )

            logger.line([val_loss],
                             [epoch],
                             opts=dict(title="Validation_loss"),
                             win='2',
                             name='valid_loss',
                             update='append'
                    )

            logger.line([val_acc],
                             [epoch],
                             opts=dict(title="Validation_Acc"),
                             win='6',
                             name='valid_acc',
                             update='append'
                    )

            logger.line([res_train_acc[-1]],
                             [epoch],
                             opts=dict(title="Training_acc"),
                             win='7',
                             name='train_acc',
                             update='append'
                    )


            # Plot confusion matrix in visdom
            # ABCDE = 01234
            logger.heatmap(train_confusion, win='4', opts=dict(
                title="Train_Confusion_Matrix_epoch_{}".format(epoch),
                columnnames=["A", "B", "C", "D", "E"],
                rownames=["A", "B", "C", "D", "E"])
                           )
    print('-'*50)
    print('save model...')
    #torch.save(model.state_dict(), settings.config["System"]["OutputFileDir"] + '\\models\\LSTM_Model.pt')
    torch.jit.script(model).save(settings.config["System"]["OutputFileDir"] + '\\models\\Jitted_LSTM_Model.pt')
    print('model was saved completely')

    #グラフ化
    traingraph(x_axis_epoch, res_train_loss, res_train_f1, res_val_loss, res_val_f1, train_confusion, val_confusion,settings)

    summary = pd.DataFrame({'Epoch': [i + 1 for i in range(epoch)], 'TrainLoss': res_train_loss, 'TrainAcc':res_train_acc,
                            'TrainPrecision':res_train_precision, 'TrainRecall': res_train_recall, 'TrainF1':res_train_f1,
                            'ValLoss':res_val_loss, 'ValAcc':res_val_acc, 'ValPrecision': res_val_precision, 'Val_Recall': res_val_recall, 'ValF1': res_val_f1})
    summary.to_csv(settings.config["System"]["OutputFileDir"]+ '\\' + str(datetime.datetime.now()).replace("-","").replace(" ","_").replace(":","").replace(".","_") +'_TrainingSummary.csv')

    config_file = open(settings.config["System"]["OutputFileDir"]+ '\\' + str(datetime.datetime.now()).replace("-","").replace(" ","_").replace(":","").replace(".","_") + "_Settings.json", "w")
    json.dump(settings.config, config_file)
    config_file.close()


    return model, res_train_loss, res_train_acc, res_val_loss, res_val_acc, res_train_f1, res_val_f1


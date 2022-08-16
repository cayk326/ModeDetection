import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix
from ModeDetection.util.tools import get_confusion_matrix
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

def validation(settings, model, val_loader, loss_fn, cur_epoch, max_epoch, logger):
    print('Start validation...')

    valid_loss = 0
    num_classes = settings.config["ModelParams"]["num_classes"]
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    confusion = np.zeros((num_classes, num_classes))
    pred_label_list = [] # 全検証データに対する推論結果を格納したリスト
    true_label_list = [] # 全検証データに対する正解ラベルを格納したリスト

    with torch.no_grad():
        model.eval()
        for iteration, (x, t) in enumerate(val_loader):
            x, t = x.to(DEVICE), t.to(DEVICE)
            y = model(x)
            valid_loss += loss_fn(y, t)# 各イタレーションの検証誤差を累積する
            rows = t.cpu().numpy() #各イタレーションの正解ラベル
            cols = y.argmax(1).cpu().numpy()# 各イタレーションの推論ラベル
            true_label_list = np.hstack((true_label_list, rows))# 各イタレーションの正解ラベルを格納
            pred_label_list = np.hstack((pred_label_list, cols))  # 各イタレーションの推論ラベルを格納


    confusion = get_confusion_matrix(true_label_list, pred_label_list, settings.config['ModelParams']['out_dim'])  # 検証データに対する混同行列の算出
    avg_valid_loss = float(valid_loss.cpu().numpy()) / iteration #累積された検証誤差をイタレーション数で割り平均を出す。検証誤差の最終値(平均値)


    acc = accuracy_score(true_label_list, pred_label_list)# acc
    recall = recall_score(true_label_list, pred_label_list, average="macro")  # Recall
    precision = precision_score(true_label_list, pred_label_list, average="macro") # Precision
    f1 = f1_score(true_label_list, pred_label_list, average="macro")  # f1

    print("★Epoch: {ep}/{max}, val_loss: {loss}, val_acc: {acc}, "
          "val_precision: {precision}, val_recall: {recall}, val_f1: {f1} ".format(
        ep=cur_epoch,
        max=settings.config["ModelParams"]["epochs"],
        loss=avg_valid_loss,
        acc=acc,
        precision=precision,
        recall=recall,
        f1=f1
    )
    )

    return avg_valid_loss, acc, precision, recall, f1, confusion
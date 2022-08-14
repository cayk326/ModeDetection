import collections
import pandas as pd
import numpy as np

def get_confusion_matrix(truelabel, predlabel, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))  # 混同行列初期化
    for t, pred in zip(truelabel, predlabel):  # # 1エポックで検証する全イタレーションにおける混同行列を作成。データ数はx_train.__len()__と等価
        # print("TrueValue:{0} | PredValue:{1}".format(t,pred))
        confusion_matrix[int(t), int(pred)] += 1
        
    return confusion_matrix





def get_all_file_length(train_dataset_list, valid_dataset_list, test_dataset_list, settings):
    dataset_len_list = collections.OrderedDict({"train": [], "valid": [], "test": []})

    '''
    -----------------------
    データ読み込み
    -----------------------
    '''

    for train in train_dataset_list:
        train_chunks = pd.read_csv(train,
                             chunksize=10000,
                             encoding=settings.config["System"]["Encoding"],
                             sep=settings.config["System"]["Deliminator"]["CSV"],
                             header=settings.config["System"]["Header_pos"]["RT"])
        train_dataset = pd.concat((data for data in train_chunks), ignore_index=True)
        dataset_len_list["train"].append(len(train_dataset))

    for valid in valid_dataset_list:

        valid_chunks = pd.read_csv(valid,
                             chunksize=10000,
                             encoding=settings.config["System"]["Encoding"],
                             sep=settings.config["System"]["Deliminator"]["CSV"],
                             header=settings.config["System"]["Header_pos"]["RT"])
        valid_dataset = pd.concat((data for data in valid_chunks), ignore_index=True)
        dataset_len_list["valid"].append(len(valid_dataset))

    for test in test_dataset_list:

        test_chunks = pd.read_csv(test,
                             chunksize=10000,
                             encoding=settings.config["System"]["Encoding"],
                             sep=settings.config["System"]["Deliminator"]["CSV"],
                             header=settings.config["System"]["Header_pos"]["RT"])
        test_dataset = pd.concat((data for data in test_chunks), ignore_index=True)
        dataset_len_list["test"].append(len(test_dataset))

    print("Finish to calculate all file length")
    return dataset_len_list
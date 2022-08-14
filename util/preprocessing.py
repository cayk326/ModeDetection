import numpy as np
import sys
import pandas as pd
seed = 32
np.random.seed(seed)

#オーバーラップ処理
def overlapping(data, samplerate, Fs, overlap_rate):
    '''
    入力データに対してオーバーラップ処理を行う
    フレームサイズを定義してデータを切り出すと切り出しができない部分が発生する
    その際の時間も返すように設定
    スペクトログラムを表示する際に使用する

    :param data: 入力データ
    :param samplerate: サンプリングレート[Hz]
    :param Fs: フレームサイズ
    :param overlap_rate: オーバーラップレート[%]

    :return:
        :array: オーバーラップ加工されたデータ
        :N_ave:　オーバーラップ加工されたデータの個数
        :final_time: 最後に切り出したデータの時間
    '''
    Ts = len(data) / samplerate         #全データ点数
    Fc = Fs / samplerate                #フレーム周期
    x_ol = Fs * (1 - (overlap_rate/100))     #オーバーラップ時のフレームずらし幅
    N_ave = int((Ts - (Fc * (overlap_rate/100))) / (Fc * (1-(overlap_rate/100)))) #抽出するフレーム数（平均化に使うデータ個数）

    array = []      #抽出したデータを入れる空配列の定義
    df = pd.DataFrame()
    #forループでデータを抽出
    for i in range(N_ave):
        ps = int(x_ol * i)              #切り出し位置をループ毎に更新
        array.append(data.values[ps:ps+Fs:1])  #切り出し位置psからフレームサイズ分抽出して配列に追加
    final_time = (ps + Fs) / samplerate
    array = np.array(array)
    return array, N_ave, final_time #オーバーラップ抽出されたデータ配列とデータ個数を戻り値にする



def generate_batch(x_dataset_list, y_dataset_list, settings):
    print("Generate batch from imported data...")
    print('-------------------Dataset Information----------------------')
    print("Number of x dataset {0}".format(len(x_dataset_list)))
    print("Number of y dataset {0}".format(len(y_dataset_list)))
    print('------------------------------------------------------------')

    res_x_batch, res_y_batch = np.array([]), np.array([])
    flag = False
    overlap_rate = int(settings.config["FFTSetting"]["overlap"] * 100)
    delta_f = settings.config["FFTSetting"]["delta_f"]
    fs_trans = settings.config["ResampleSetting"]["Fs_trans"]
    Frame_size = int(fs_trans / delta_f)

    '''
    MaxMinVal = {'Time':120, 'MA_deg':180, 'AccelPosition_percent': 100, 'BrakeStroke_mm':-30,
                 'VehicleSpeed_kph':240, 'GX':15, 'GY':15, 'Mileage_m':5}
    '''

    for x_data_path, y_data_path in zip(x_dataset_list, y_dataset_list):
        x_chunks = pd.read_csv(x_data_path,
                               chunksize=10000,
                               encoding=settings.config["System"]["Encoding"],
                               sep=settings.config["System"]["Deliminator"]["CSV"],
                               header=settings.config["System"]["Header_pos"]["Common"])
        x_data = pd.concat((data for data in x_chunks), ignore_index=True)
        #x_data.drop(columns=settings.config["Prepro"]["drop_label"], inplace=True)# データのドロップ
        x_data = x_data[settings.config["Prepro"]["use_label"]]
        print("x data frame memory_size:{0}".format(sys.getsizeof(x_data) / 1000000) + "[MB]")
        '''
        #-------正規化------#
        for column in x_data.columns:
            if MaxMinVal[column] != None:
                x_data[column] = x_data[column] / MaxMinVal[column]
        '''

        y_chunks = pd.read_csv(y_data_path,
                               chunksize=10000,
                               encoding=settings.config["System"]["Encoding"],
                               sep=settings.config["System"]["Deliminator"]["CSV"],
                               header=settings.config["System"]["Header_pos"]["Common"])
        y_data = pd.concat((data for data in y_chunks), ignore_index=True)
        print("y data frame memory_size:{0}".format(sys.getsizeof(y_data) / 1000000) + "[MB]")
        print("Label : " + str(np.unique(y_data)))
        # x_batch, N_x_batch, final_time_x_batch = overlapping(x_data, 10, int(10/0.1), 50)
        # batch => バッチサイズ、シーケンス長、特徴量
        x_batch, N_x_batch, final_time_x_batch = overlapping(x_data, settings.config["ResampleSetting"]["Fs_trans"], Frame_size, overlap_rate)#フレームサイズ=samplerate/delta_f
        print('create one x batch')
        # batch => バッチサイズ、シーケンス長、正解ラベル(1次元)
        y_batch, N_y_batch, final_time_y_batch = overlapping(y_data, settings.config["ResampleSetting"]["Fs_trans"], Frame_size, overlap_rate)
        print('create one y batch')

        if flag == False:
            res_x_batch = x_batch.transpose(1, 2, 0)# シーケンス長、特徴量、バッチサイズに変換
            flag = True
        else:
            res_x_batch = np.block([res_x_batch, x_batch.transpose(1, 2, 0)])
        #res_y_batch = np.append(res_y_batch, np.full(N_y_batch, int(y_batch[:].mean())))
        res_y_batch = np.append(res_y_batch, np.array([np.median(y_batch[i].flatten()) for i in range(len(y_batch))]))
    res_x_batch = res_x_batch.transpose(2,0,1)
    return res_x_batch, res_y_batch



class batch_norm:
    def fit_transform(self, x_batch):
        mu_scale, std_scale = self.fit(x_batch)
        fit_x = self.transform(x_batch, mu_scale, std_scale)
        return fit_x, mu_scale, std_scale

    def fit(self, x_batch):
        mu_scale = np.array([x_batch[:,:,i].mean() for i in range(x_batch.shape[2])])# バッチ行×シーケンス長の行列に対して全体の平均を求める。この処理を各特徴量に実施
        std_scale = np.array([x_batch[:,:,i].std() for i in range(x_batch.shape[2])])# バッチ行×シーケンス長の行列に対して全体の標準偏差を求める。この処理を各特徴量に実施
        return mu_scale, std_scale

    def transform(self, x_batch, mu_scale, std_scale):
        fit_x = (x_batch - mu_scale) * (1 / std_scale)

        return fit_x

def numerical_gradient(x, dt):
    """

    :param x:  data
    :param dt: sampling time[s]
    :return:
    """
    print("Calc numerical gradient")
    from collections import deque
    diff = deque(np.float64(np.float64(x[1:]) - np.float64(x[0:-1])))# Backward Difference f(x + h) - f(x).
    diff.appendleft(diff[0])
    grad = diff / (dt + 1e-8)
    return grad

def calc_Jerk(Gx, Gy, Time):
    """
    Calculate Jerk using Gx and Gy

    :param Gx: unit is g
    :param Gy: unit is g
    :return: comp_jerk, jerk_x, jerk_y
    """
    print("Calc Jerk")
    dt = Time[1] - Time[0]
    Jerk_x = numerical_gradient(Gx, dt)
    Jerk_y = numerical_gradient(Gy, dt)
    J = np.sqrt(Jerk_x ** 2 + Jerk_y ** 2)
    return J, Jerk_x, Jerk_y



import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np
import os

import random

seed = 32
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(seed)

class LSTMwithFC(nn.Module):

    def __init__(self, feature_size=28, hidden_dim=128, num_layers=1, out_dim=5, dropout_ratio=0, classification=True, batch_first=True):
        super(LSTMwithFC, self).__init__()
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.feature_size = feature_size  # The number of expected features in the input x. Number of features you want to use.
        self.hidden_layer_size = hidden_dim  # The number of features in the hidden state h. Vector size of each hidden layer.
        self.num_layers = num_layers  # The number of reccurent layers
        self.out_dim = out_dim # The number of expected output dimension of fully connected network.
        self.dropout_ratio = dropout_ratio # dropout ratio
        self.classification = classification # classification flag
        self.batch_first = batch_first # batch first flag

        self.hidden_0 = torch.zeros(self.num_layers, 1, self.hidden_layer_size) # hidden state
        self.cell_0 = torch.zeros(self.num_layers, 1, self.hidden_layer_size) # cell state

        self.lstm = nn.LSTM(input_size=self.feature_size,
                            hidden_size=self.hidden_layer_size,
                            num_layers=self.num_layers,
                            batch_first=self.batch_first,
                            dropout=self.dropout_ratio)


        self.fc = nn.Linear(self.hidden_layer_size, self.out_dim)
        self.softmax = nn.Softmax(dim=1)# 列方向にsoftmaxをかける


    def forward(self, x):
        batch_size = x.shape[0]
        self.hidden_0 = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(self.DEVICE)# 隠れ状態の初期化
        self.cell_0 = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(self.DEVICE)# 記憶セルの初期化

        lstm_out, (h_n, c_n) = self.lstm(x, (self.hidden_0, self.cell_0))# LSTM

        last_hidden_layer = h_n[-1, :, :]  # lstm_layersの最後のレイヤーを取り出す  (B, h)

        out = lstm_out[:, -1, :]
        out = self.fc(out)# Dense
        if self.classification:# 分類問題の時は確率に落とし込む
            out = self.softmax(out)# Softmax
        return out

if __name__ == '__main__':
    seq_len = 100
    feature_size = 5
    batch_size = 16
    model = LSTMwithFC(
                 feature_size=5,
                 hidden_dim=32,
                 num_layers=2,
                 out_dim=3,
                 dropout_ratio=0.1,
                 classification=True
                 )
    x = torch.rand(batch_size, seq_len, feature_size)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(model)
    model.eval()
    model.to(DEVICE)
    x.to(DEVICE)
    jit_model = torch.jit.script(model)
    print(jit_model)
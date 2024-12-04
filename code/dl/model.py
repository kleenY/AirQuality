# -*- coding: utf-8 -*
# @Time : 2024/11/16 21:38
# @Author : 杨坤林
# @File : model.py
# @Software : PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore

import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.arima.model import ARIMA



# LSTM 模型构建和训练使用 PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出
        return out


class my_LSTM(nn.Module):
    def __init__(self,
                 input_size=12,
                 hidden_size=256,
                 num_layers=2,
                 output_size=1,
                 dropout=0.2,
                 seq_len=24,
                 pred_len=6,
                 device=torch.device('cuda:0')
                 ):
        super(my_LSTM, self).__init__()
        self.outout_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_size = input_size
        self.device = device
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.batch_first = True
        self.bidirectional = False
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, dropout=self.dropout, batch_first=self.batch_first,
                            bidirectional=self.bidirectional)

        self.reg = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                 nn.Tanh(),
                                 nn.Linear(self.hidden_size, self.outout_size),
                                 )
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def initial_hidden_state(self, batch_size):
        '''(num_layers * num_directions, batch_size, hidden_size )'''
        if self.bidirectional == False:
            num_directions = 1
        else:
            num_directions = 2
        h_0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers * num_directions, batch_size, self.hidden_size).to(self.device)
        # print(num_directions)
        hidden_state = (h_0, c_0)
        return hidden_state

    def forward(self, x):
        hidden_state = self.initial_hidden_state(x.size(0))
        lstm_out, hidden = self.lstm(x, hidden_state)
        outputs = self.reg(lstm_out)
        # print(outputs.shape)
        outputs = self.Linear(outputs.permute(0, 2, 1)).permute(0, 2, 1)

        return outputs


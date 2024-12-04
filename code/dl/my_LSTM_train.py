# -*- coding: utf-8 -*
# @Time : 2024/11/22 18:16
# @Author : 杨坤林
# @File : my_LSTM_train.py
# @Software : PyCharm
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tools import *
from tools2 import *
from model import *



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
                                 nn.ReLU(),
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






def my_train_3(pkl_file_path, output_dir='LSTM_output', step=1, weight_decay=1e-4, target_column = 'CO(GT)'):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 选择输入特征和目标变量
    selected_features = ['PT08.S5(O3)', 'PT08.S4(NO2)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S1(CO)', target_column]
    # target_column = 'CO(GT)'
    # 训练模型
    num_epochs = 200

    # 数据加载与预处理
    with open(pkl_file_path, 'rb') as file:
        data_list = pickle.load(file)

    # 提取输入和输出数据
    train_X = []
    train_y = []

    for sample in data_list:
        input_df, output_df = sample
        input_df = input_df[selected_features]  # 选择特定特征
        train_X.append(input_df.values)
        train_y.append(output_df[target_column])  # 确保输出形状正确

    train_X = np.array(train_X)
    train_y = np.array(train_y)

    # 获取序列长度和预测长度
    seq_len = train_X.shape[1]
    pred_len = train_y.shape[1]

    # 归一化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_X = scaler_X.fit_transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    train_y = scaler_y.fit_transform(train_y.reshape(-1, train_y.shape[-1])).reshape(train_y.shape)

    # 转换为 PyTorch 张量
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y.squeeze(), dtype=torch.float32)  # 将 y 的形状简化为 [batch_size, future_hours]

    # 划分验证集
    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.05, shuffle=False)

    # 构建 DataLoader
    batch_size = 32
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    # 模型参数
    input_size = len(selected_features)
    hidden_size = 64
    output_size = 1

    # 训练和测试函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = my_LSTM(input_size=input_size, output_size=output_size, seq_len=seq_len, pred_len=pred_len, device=device).to(device)
    # criterion = nn.functional.smooth_l1_loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)  # 加入正则化

    best_val_mae = float('inf')
    model_save_path = os.path.join(output_dir, 'best_LSTM_model.pth')

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = nn.functional.smooth_l1_loss(outputs.squeeze(), batch_y)
            # loss = criterion(outputs.squeeze(), batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证集损失计算
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = nn.functional.smooth_l1_loss(outputs.squeeze(), batch_y)
                # loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # # 每隔几个 epoch 保存一次模型
        # if (epoch + 1) % 100 == 0:
        #     epoch_model_path = os.path.join(output_dir, f'LSTM_model_epoch_{epoch+1}.pth')
        #     torch.save(model.state_dict(), epoch_model_path)
        #     print(f"Model saved to {epoch_model_path}")

        # 如果验证集 MAE 更好，则保存最佳模型


        if val_loss < best_val_mae:
            best_val_mae = val_loss
            try:
                torch.save(model.state_dict(), model_save_path)
                print(f"Best model saved to {model_save_path}")
            except Exception as e:
                print(f"Failed to save model to {model_save_path}. Error: {e}")

    # 保存训练和验证损失到 CSV 文件
    losses_df = pd.DataFrame({'Epoch': range(1, num_epochs + 1), 'Train Loss': train_losses, 'Validation Loss': val_losses})
    losses_csv_path = os.path.join(output_dir, 'LSTM_losses.csv')
    losses_df.to_csv(losses_csv_path, index=False)
    print(f"Losses saved to {losses_csv_path}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("LSTM Train and Validation Loss Over Epochs")
    plt.legend()
    loss_plot_path = os.path.join(output_dir, "LSTM_train_val_loss.png")
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")
    plt.close()

    # 保存标准化器对象
    scaler_X_path = os.path.join(output_dir, 'scaler_X.pkl')
    scaler_y_path = os.path.join(output_dir, 'scaler_y.pkl')
    with open(scaler_X_path, 'wb') as file:
        pickle.dump(scaler_X, file)
    with open(scaler_y_path, 'wb') as file:
        pickle.dump(scaler_y, file)
    print(f"Scalers saved to {scaler_X_path} and {scaler_y_path}")








if __name__ == '__main__':
    #
    # pkl_file_path = '..\\data\\split_data\\long_train.pkl'
    # target_column = 'CO(GT)'
    # my_train_3(pkl_file_path=pkl_file_path, output_dir =target_column +'_long_LSTM_output', target_column=target_column)



    # 定义数据类型和目标列
    data_types = ['short', 'medium', 'long']
    target_columns = ['CO(GT)', 'NOx(GT)', 'NO2(GT)']

    # 循环训练
    for data_type in data_types:
        for target_column in target_columns:
            pkl_file_path = f'..\\data\\split_data\\{data_type}_train.pkl'
            output_dir = f'{target_column}_{data_type}_LSTM_output'
            my_train_3(pkl_file_path=pkl_file_path, output_dir=output_dir, target_column=target_column)












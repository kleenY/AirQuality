# -*- coding: utf-8 -*
# @Time : 2024/11/10 0:07
# @Author : 杨坤林
# @File : my_main.py
# @Software : PyCharm


from tools import *
from tools2 import *
from model import *

file_path = '..\\data\\AirQualityUCI.xlsx'


# 2. 特征和标签生成函数
def create_features_and_labels(data, past_hours, future_hours):
    X, y = [], []
    for i in range(len(data) - past_hours - future_hours + 1):
        X.append(data[i:i + past_hours])
        y.append(data[i + past_hours:i + past_hours + future_hours])
    return np.array(X), np.array(y)


def main():
    # # 1. 数据加载与预处理
    # data = pd.read_csv("air_quality.csv", index_col=0, parse_dates=True)
    # data = data.interpolate(method='time')  # 填补缺失值


    data = data_process()


    # 选择输入特征和目标变量
    target_column = 'CO(GT)'
    input_features = [col for col in data.columns if col != target_column]

    # 特征缩放
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaled_features = scaler_X.fit_transform(data[input_features])
    scaled_target = scaler_y.fit_transform(data[[target_column]])

    # 创建输入输出数据
    def create_features_and_labels(data_X, data_y, past_hours, future_hours):
        X, y = [], []
        for i in range(len(data_X) - past_hours - future_hours + 1):
            X.append(data_X[i:i + past_hours])
            y.append(data_y[i + past_hours:i + past_hours + future_hours])
        return np.array(X), np.array(y)

    # 参数设置
    past_hours = 24  # 输入的历史小时数
    future_hours = 6  # 输出的预测小时数

    X, y = create_features_and_labels(scaled_features, scaled_target, past_hours, future_hours)

    # 转换为 PyTorch 张量
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.squeeze(), dtype=torch.float32)  # 将 y 的形状简化为 [batch_size, future_hours]

    # 滚动窗口参数
    window_size = 30 * 24  # 滚动窗口大小：30 天
    stride = 7 * 24  # 滑动步长：7 天
    num_windows = (len(X) - window_size) // stride



    # 模型参数
    input_size = X.shape[2]  # 输入特征的数量
    hidden_size = 64
    output_size = 1  # 输出未来几个小时的预测值

    # 3. 滚动窗口验证
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_and_evaluate_window(X, y, window_start, window_size, stride):
        # 定义训练和验证集
        train_X = X[:window_start + window_size]
        train_y = y[:window_start + window_size]
        val_X = X[window_start + window_size:window_start + window_size + stride]
        val_y = y[window_start + window_size:window_start + window_size + stride]

        # 构建 DataLoader
        batch_size = 32
        train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=batch_size)

        # 定义模型、损失函数和优化器
        model = my_LSTM().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        num_epochs = 1
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                batch_y = batch_y.unsqueeze(-1)

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            train_loss /= len(train_loader)

        # 验证模型
        model.eval()
        val_loss = 0
        y_pred, y_true = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                batch_y = batch_y.unsqueeze(-1)
                val_loss += criterion(outputs, batch_y).item()
                y_pred.append(outputs.cpu().numpy())
                y_true.append(batch_y.cpu().numpy())
        val_loss /= len(val_loader)
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        return model, train_loss, val_loss, y_pred, y_true

    # 开始滚动窗口验证
    overall_mae = []
    for window_idx in range(2):
        window_start = window_idx * stride
        model, train_loss, val_loss, y_pred, y_true = train_and_evaluate_window(
            X, y, window_start, window_size, stride
        )
        # 反归一化后计算 MAE
        y_pred_rescaled = scaler_y.inverse_transform(y_pred.squeeze())
        y_true_rescaled = scaler_y.inverse_transform(y_true.squeeze())
        mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
        overall_mae.append(mae)
        print(
            f"Window {window_idx + 1}/{num_windows}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, MAE: {mae:.4f}")

    # 打印整体性能
    print(f"Overall MAE (mean of all windows): {np.mean(overall_mae):.4f}")

    # 选择一个样本进行可视化（这里选取第一个样本）
    sample_idx = 0

    # 创建时间步（这里是6个小时）
    time_steps = np.arange(1, future_hours + 1)

    # 可视化真实值和预测值
    plt.figure(figsize=(10, 6))

    # 真实值：第一个样本的真实数据
    plt.plot(time_steps, y_true_rescaled[sample_idx, :], label="True", marker='o')

    # 预测值：第一个样本的预测数据
    plt.plot(time_steps, y_pred_rescaled[sample_idx, :], label="Predicted", marker='x')

    # 添加图例和标题
    plt.legend()
    plt.title("Air Quality Prediction (Sample 1)")
    plt.xlabel("Hours Ahead")
    plt.ylabel("CO Concentration (Rescaled)")
    plt.xticks(time_steps)  # 确保横坐标显示小时数

    # 显示图像
    plt.show()


def main_2():

    # 1. 数据加载与预处理
    # data = pd.read_csv("air_quality.csv", index_col=0, parse_dates=True)
    # data = data.interpolate(method='time')  # 填补缺失值

    data = data_process()

    # 选择输入特征和目标变量
    target_column = 'CO(GT)'
    input_features = [col for col in data.columns if col != target_column]

    # 特征缩放
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaled_features = scaler_X.fit_transform(data[input_features])
    scaled_target = scaler_y.fit_transform(data[[target_column]])

    # 创建输入输出数据
    def create_features_and_labels(data_X, data_y, past_hours, future_hours):
        X, y = [], []
        for i in range(len(data_X) - past_hours - future_hours + 1):
            X.append(data_X[i:i + past_hours])
            y.append(data_y[i + past_hours:i + past_hours + future_hours])
        return np.array(X), np.array(y)

    # 参数设置
    past_hours = 24  # 输入的历史小时数
    future_hours = 6  # 输出的预测小时数

    X, y = create_features_and_labels(scaled_features, scaled_target, past_hours, future_hours)

    # 转换为 PyTorch 张量
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.squeeze(), dtype=torch.float32)  # 将 y 的形状简化为 [batch_size, future_hours]

    # 2. 数据拆分：80% 用于训练，20% 用于测试
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 构建 DataLoader
    batch_size = 32
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    # 模型参数
    input_size = X.shape[2]  # 输入特征的数量
    hidden_size = 64
    output_size = 1  # 输出未来几个小时的预测值

    # 训练和测试函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = my_LSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 1  # 设置训练轮数
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            batch_y = batch_y.unsqueeze(-1)

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")

    # 测试模型
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            batch_y = batch_y.unsqueeze(-1)
            y_pred.append(outputs.cpu().numpy())
            y_true.append(batch_y.cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    # 反归一化
    y_pred_rescaled = scaler_y.inverse_transform(y_pred.squeeze())
    y_true_rescaled = scaler_y.inverse_transform(y_true.squeeze())

    # 计算 MAE
    mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
    print(f"Test MAE: {mae:.4f}")

    # 3. 保存模型
    model_save_path = 'lstm_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # 4. 可视化最后一个样本的结果
    sample_idx = 0
    time_steps = np.arange(1, future_hours + 1)

    # 可视化真实值和预测值
    plt.figure(figsize=(10, 6))

    # 真实值：第一个样本的真实数据
    plt.plot(time_steps, y_true_rescaled[sample_idx, :, 0], label="True", marker='o')

    # 预测值：第一个样本的预测数据
    plt.plot(time_steps, y_pred_rescaled[sample_idx, :, 0], label="Predicted", marker='x')

    # 添加图例和标题
    plt.legend()
    plt.title("Air Quality Prediction (Sample 1)")
    plt.xlabel("Hours Ahead")
    plt.ylabel("CO Concentration (Rescaled)")
    plt.xticks(time_steps)  # 确保横坐标显示小时数

    # 保存图像
    plt.savefig('prediction_sample_1.png')

    # 显示图像
    plt.show()


import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

def my_train_3(output_dir='LSTM_output', step=1, weight_decay=1e-4):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 选择输入特征和目标变量
    target_column = 'CO(GT)'

    # 数据加载与预处理
    data = data_process(target_column)
    input_features = [col for col in data.columns if col != target_column]

    # 特征缩放
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaled_features = scaler_X.fit_transform(data[input_features])
    scaled_target = scaler_y.fit_transform(data[[target_column]])

    # 创建输入输出数据
    def create_features_and_labels(data_X, data_y, past_hours, future_hours, step):
        X, y = [], []
        for i in range(0, len(data_X) - past_hours - future_hours + 1, step):
            X.append(data_X[i:i + past_hours])
            y.append(data_y[i + past_hours:i + past_hours + future_hours])
        return np.array(X), np.array(y)

    # 参数设置
    past_hours = 24  # 输入的历史小时数
    future_hours = 6  # 输出的预测小时数

    X, y = create_features_and_labels(scaled_features, scaled_target, past_hours, future_hours, step)

    # 转换为 PyTorch 张量
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.squeeze(), dtype=torch.float32)  # 将 y 的形状简化为 [batch_size, future_hours]

    # 数据拆分
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    # 构建 DataLoader
    batch_size = 32
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    # 模型参数
    input_size = X.shape[2]
    hidden_size = 64
    output_size = 1

    # 训练和测试函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = my_LSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)  # 加入正则化

    # 训练模型
    num_epochs = 100
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
            batch_y = batch_y.unsqueeze(-1)

            loss = criterion(outputs, batch_y)
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
                batch_y = batch_y.unsqueeze(-1)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 每隔几个 epoch 保存一次模型
        if (epoch + 1) % 5 == 0:
            epoch_model_path = os.path.join(output_dir, f'LSTM_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), epoch_model_path)
            print(f"Model saved to {epoch_model_path}")

        # 如果验证集 MAE 更好，则保存最佳模型
        if val_loss < best_val_mae:
            best_val_mae = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved to {model_save_path}")

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





# 加载模型并进行评估
import os  # 用于创建和管理文件夹

def my_test_model(model_save_path, output_dir="LSTM_results", batch_size=32):
    """
    Function to test the model on the test dataset.

    Parameters:
    - model_save_path: Path to the saved model
    - output_dir: Directory to save all output files
    - batch_size: Batch size for DataLoader

    Saves:
    - CSV file with evaluation metrics: MAE, RMSE, PMAE, PRMSE
    - Prediction plot for a test sample
    """

    # 创建保存结果的文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 选择输入特征和目标变量
    target_column = 'CO(GT)'
    # 数据加载与预处理
    data = data_process(target_column)

    input_features = [col for col in data.columns if col != target_column]

    # 特征缩放
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaled_features = scaler_X.fit_transform(data[input_features])
    scaled_target = scaler_y.fit_transform(data[[target_column]])

    # 创建输入输出数据
    def create_features_and_labels(data_X, data_y, past_hours, future_hours):
        X, y = [], []
        for i in range(len(data_X) - past_hours - future_hours + 1):
            X.append(data_X[i:i + past_hours])
            y.append(data_y[i + past_hours:i + past_hours + future_hours])
        return np.array(X), np.array(y)

    # 参数设置
    past_hours = 24  # 输入的历史小时数
    future_hours = 6  # 输出的预测小时数

    X, y = create_features_and_labels(scaled_features, scaled_target, past_hours, future_hours)

    # 转换为 PyTorch 张量
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.squeeze(), dtype=torch.float32)  # 将 y 的形状简化为 [batch_size, future_hours]

    # 数据拆分：80% 用于训练，10% 用于验证，10% 用于测试
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 构建测试集的 DataLoader
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = my_LSTM().to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()  # 切换到评估模式

    # 预测测试集
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            batch_y = batch_y.unsqueeze(-1)
            y_pred.append(outputs.cpu().numpy())
            y_true.append(batch_y.cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    # 反归一化
    y_pred_rescaled = scaler_y.inverse_transform(y_pred.squeeze())
    y_true_rescaled = scaler_y.inverse_transform(y_true.squeeze())

    # 计算评估指标
    mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
    rmse = mean_squared_error(y_true_rescaled, y_pred_rescaled, squared=False)
    pmae = mae / np.mean(y_true_rescaled) * 100
    prmse = rmse / np.mean(y_true_rescaled) * 100

    # 打印结果
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test PMAE: {pmae:.2f}%")
    print(f"Test PRMSE: {prmse:.2f}%")

    # 保存结果到 CSV 文件
    results = {
        "Metric": ["MAE", "RMSE", "PMAE (%)", "PRMSE (%)"],
        "Value": [mae, rmse, pmae, prmse]
    }
    results_df = pd.DataFrame(results)
    metrics_path = os.path.join(output_dir, "LSTM_test_metrics.csv")
    results_df.to_csv(metrics_path, index=False)
    print(f"Test metrics saved to '{metrics_path}'")

    # 可视化最后一个样本的结果
    sample_idx = 0
    time_steps = np.arange(1, future_hours + 1)

    plt.figure(figsize=(10, 6))

    # 真实值：第一个样本的真实数据
    plt.plot(time_steps, y_true_rescaled[sample_idx, :], label="True", marker='o')

    # 预测值：第一个样本的预测数据
    plt.plot(time_steps, y_pred_rescaled[sample_idx, :], label="Predicted", marker='x')

    # 添加图例和标题
    plt.legend()
    plt.title("Air Quality Prediction (LSTM Test Sample 1)")
    plt.xlabel("Hours Ahead")
    plt.ylabel("CO Concentration (Rescaled)")
    plt.xticks(time_steps)  # 确保横坐标显示小时数

    # 保存图像
    plot_path = os.path.join(output_dir, 'LSTM_prediction_test_sample_1.png')
    plt.savefig(plot_path)
    print(f"Prediction plot saved to '{plot_path}'")

    return mae


def my_test_model_long_term(model_save_path, output_dir="LSTM_results", batch_size=32):
    """
    Test the model and visualize predictions over a longer continuous time period without overlap.

    Parameters:
    - model_save_path: Path to the saved model
    - output_dir: Directory to save all output files
    - batch_size: Batch size for DataLoader

    Saves:
    - CSV file with evaluation metrics: MAE, RMSE, PMAE, PRMSE
    - Prediction plot with continuous time series
    """

    import os  # 用于创建和管理文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 选择输入特征和目标变量
    target_column = 'CO(GT)'
    # 数据加载与预处理
    data = data_process(target_column)

    input_features = [col for col in data.columns if col != target_column]

    # 特征缩放
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaled_features = scaler_X.fit_transform(data[input_features])
    scaled_target = scaler_y.fit_transform(data[[target_column]])

    # 创建输入输出数据
    def create_features_and_labels(data_X, data_y, past_hours, future_hours, step):
        X, y = [], []
        for i in range(0, len(data_X) - past_hours - future_hours + 1, step):
            X.append(data_X[i:i + past_hours])
            y.append(data_y[i + past_hours:i + past_hours + future_hours])
        return np.array(X), np.array(y)

    # 参数设置
    past_hours = 24  # 输入的历史小时数
    future_hours = 6  # 输出的预测小时数
    step = future_hours  # 滑动窗口步长

    X, y = create_features_and_labels(scaled_features, scaled_target, past_hours, future_hours, step)

    # 转换为 PyTorch 张量
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.squeeze(), dtype=torch.float32)  # 将 y 的形状简化为 [batch_size, future_hours]

    # 数据拆分：80% 用于训练，10% 用于验证，10% 用于测试
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 构建测试集的 DataLoader
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = my_LSTM().to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()  # 切换到评估模式

    # 预测测试集
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            y_pred.append(outputs.cpu().numpy())
            y_true.append(batch_y.cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    # 反归一化
    y_pred_rescaled = scaler_y.inverse_transform(y_pred.squeeze())
    y_true_rescaled = scaler_y.inverse_transform(y_true.squeeze())

    # 拼接连续时间序列
    predicted_series = y_pred_rescaled.flatten()
    true_series = y_true_rescaled.flatten()

    # 计算评估指标
    mae = mean_absolute_error(true_series, predicted_series)
    rmse = mean_squared_error(true_series, predicted_series, squared=False)
    pmae = mae / np.mean(true_series) * 100
    prmse = rmse / np.mean(true_series) * 100

    # 打印结果
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test PMAE: {pmae:.2f}%")
    print(f"Test PRMSE: {prmse:.2f}%")

    # 保存结果到 CSV 文件
    results = {
        "Metric": ["MAE", "RMSE", "PMAE (%)", "PRMSE (%)"],
        "Value": [mae, rmse, pmae, prmse]
    }
    results_df = pd.DataFrame(results)
    metrics_path = os.path.join(output_dir, "LSTM_test_metrics_long_term.csv")
    results_df.to_csv(metrics_path, index=False)
    print(f"Test metrics saved to '{metrics_path}'")

    # 可视化长时间序列
    plt.figure(figsize=(12, 6))

    time_steps = np.arange(1, len(true_series) + 1)
    plt.plot(time_steps, true_series, label="True", color='blue')
    plt.plot(time_steps, predicted_series, label="Predicted", color='orange')

    plt.legend()
    plt.title("Air Quality Prediction (Continuous Time Series)")
    plt.xlabel("Time Steps")
    plt.ylabel("CO Concentration (Rescaled)")

    # 保存图像
    plot_path = os.path.join(output_dir, 'LSTM_prediction_continuous.png')
    plt.savefig(plot_path)
    print(f"Prediction plot saved to '{plot_path}'")

    return mae

def my_test_model_long_term_recursive(model_save_path, output_dir="LSTM_results", past_hours=24, future_steps=20):
    """
    Test the model using rolling (recursive) forecasting for long-term predictions.

    Parameters:
    - model_save_path: Path to the saved model
    - output_dir: Directory to save all output files
    - past_hours: Number of past hours used as input
    - future_steps: Number of time steps to predict in the future

    Saves:
    - CSV file with evaluation metrics
    - Prediction plot with continuous time series
    """

    import os  # 用于创建和管理文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 选择输入特征和目标变量
    target_column = 'CO(GT)'
    # 数据加载与预处理
    data = data_process(target_column)

    input_features = [col for col in data.columns if col != target_column]

    # 特征缩放
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaled_features = scaler_X.fit_transform(data[input_features])
    scaled_target = scaler_y.fit_transform(data[[target_column]])

    # 获取最后的历史数据作为初始输入
    X_init = scaled_features[-past_hours:]  # 初始的输入特征
    true_values = scaled_target[-future_steps:]  # 真值目标值（对比用）

    # 转换为 PyTorch 张量
    X_init = torch.tensor(X_init, dtype=torch.float32).unsqueeze(0)  # 添加 batch 维度
    true_values = torch.tensor(true_values, dtype=torch.float32).squeeze()

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = my_LSTM().to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()  # 切换到评估模式

    # 滚动预测
    y_pred = []
    X_current = X_init.to(device)

    with torch.no_grad():
        for step in range(future_steps):
            # 模型预测
            output = model(X_current)

            # 取出第一个预测值，假设是预测目标值
            predicted_value = output[:, 0].cpu().numpy().flatten()  # 获取未来第一个小时的预测值

            # 将目标变量的预测值放到剩余特征中
            previous_features = X_current[:, -1, :].cpu().numpy()  # 获取当前时间步的所有特征
            previous_features[:, 0] = predicted_value  # 用预测值替换目标变量

            # 将更新后的特征归一化，生成新的输入
            new_input_scaled = scaler_X.transform(previous_features)
            new_input_scaled = torch.tensor(new_input_scaled, dtype=torch.float32).to(device)

            # 滚动窗口更新
            X_current = torch.cat((X_current[:, 1:, :], new_input_scaled.unsqueeze(1)), dim=1)

    # 反归一化
    y_pred_rescaled = scaler_y.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
    true_values_rescaled = scaler_y.inverse_transform(true_values.numpy().reshape(-1, 1)).flatten()

    # 计算评估指标
    mae = mean_absolute_error(true_values_rescaled, y_pred_rescaled)
    rmse = mean_squared_error(true_values_rescaled, y_pred_rescaled, squared=False)
    pmae = mae / np.mean(true_values_rescaled) * 100
    prmse = rmse / np.mean(true_values_rescaled) * 100

    # 打印结果
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test PMAE: {pmae:.2f}%")
    print(f"Test PRMSE: {prmse:.2f}%")

    # 保存结果到 CSV 文件
    results = {
        "Metric": ["MAE", "RMSE", "PMAE (%)", "PRMSE (%)"],
        "Value": [mae, rmse, pmae, prmse]
    }
    results_df = pd.DataFrame(results)
    metrics_path = os.path.join(output_dir, "LSTM_test_metrics_recursive.csv")
    results_df.to_csv(metrics_path, index=False)
    print(f"Test metrics saved to '{metrics_path}'")

    # 可视化滚动预测
    plt.figure(figsize=(12, 6))

    time_steps = np.arange(1, future_steps + 1)
    plt.plot(time_steps, true_values_rescaled, label="True", color='blue')
    plt.plot(time_steps, y_pred_rescaled, label="Predicted", color='orange')

    plt.legend()
    plt.title("Air Quality Prediction (Rolling Forecast)")
    plt.xlabel("Time Steps")
    plt.ylabel("CO Concentration (Rescaled)")

    # 保存图像
    plot_path = os.path.join(output_dir, 'LSTM_prediction_recursive.png')
    plt.savefig(plot_path)
    print(f"Prediction plot saved to '{plot_path}'")
    plt.close()

    return mae



def my_test():
    # 调用示例
    # 假设你已经训练完模型并保存了它
    model_save_path = './LSTM_output/best_LSTM_model.pth'
    mae = my_test_model(model_save_path)
    mae1 = my_test_model_long_term(model_save_path)
    # mae = my_test_model_long_term_recursive(model_save_path)
    print(mae)




def data_process(target = 'CO(GT)'):
    columns_to_correct = [
        'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
        'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
        'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
        'T', 'RH', 'AH'
    ]

    # 数据可视化
    # plot_air_quality_time_series_with_outlier_removal(file_path, columns_to_correct)
    # 数据预处理
    new_data = process_air_quality_data(file_path, columns_to_correct)

    # GitHub 数据集划分
    # X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(new_data, columns_to_correct)
    # train_neural_network_pytorch

    # 特征工程
    # 假设目标列名为 'CO(GT)'
    # 1. 特征提取
    # new_data = feature_extraction_show(new_data)
    new_data = feature_extraction(new_data)
    # 2. 特征选择和PCA降维
    data_pca, _ = feature_selection_show(new_data, target_column=target, save_folder='NO2(GT)_PCA')
    # data_pca, target = feature_selection(new_data, target_column='CO(GT)', n_components=5)
    # 3. 在PCA降维后进行交互特征生成
    # cdata_with_interactions, _ = create_interaction_features_from_pca(data_pca, target_column=target,
    #                                                                        data=new_data)
    #
    # return cdata_with_interactions


def data_process_save(target='CO(GT)', save_path="processed_data.csv"):
    """
    数据处理并保存为 CSV 文件。

    Args:
        target (str): 目标列名，例如 'CO(GT)'。
        save_path (str): 保存处理后数据的 CSV 文件路径，默认保存为 'processed_data.csv'。

    Returns:
        None
    """
    columns_to_correct = [
        'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
        'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
        'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
        'T', 'RH', 'AH'
    ]

    # 数据预处理
    new_data = process_air_quality_data(file_path, columns_to_correct)

    # 特征工程
    # 1. 特征提取
    new_data = feature_extraction(new_data)
    # 2. 特征选择和PCA降维
    data_pca, _ = feature_selection_show(new_data, target_column=target)
    # 3. 在PCA降维后进行交互特征生成
    cdata_with_interactions, _ = create_interaction_features_from_pca(
        data_pca, target_column=target, data=new_data
    )
    cdata_with_interactions = pd.concat([cdata_with_interactions, new_data['timestamp']], axis=1)
    # 将处理后的数据保存为 CSV 文件
    cdata_with_interactions.to_csv(save_path, index=False)

    print(f"处理后的数据已保存到: {save_path}")

def data_process_test(test_file_path= '',  target='CO(GT)', save_path="processed_data.csv"):
    """
    数据处理并保存为 CSV 文件。

    Args:
        target (str): 目标列名，例如 'CO(GT)'。
        save_path (str): 保存处理后数据的 CSV 文件路径，默认保存为 'processed_data.csv'。

    Returns:
        None
    """
    columns_to_correct = [
        'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
        'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
        'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
        'T', 'RH', 'AH'
    ]

    # 数据预处理
    new_data = process_air_quality_data(test_file_path, columns_to_correct)

    # 特征工程
    # 1. 特征提取
    new_data = feature_extraction(new_data)
    # 2. 特征选择和PCA降维
    path = target + '_pca_model.joblib'
    data_pca, _ = feature_selection_test(new_data, target_column=target, load_path= path)
    # 3. 在PCA降维后进行交互特征生成
    cdata_with_interactions, _ = create_interaction_features_from_pca(
        data_pca, target_column=target, data=new_data
    )
    cdata_with_interactions = pd.concat([cdata_with_interactions, new_data['timestamp']], axis=1)
    # 将处理后的数据保存为 CSV 文件
    cdata_with_interactions.to_csv(save_path, index=False)

    print(f"处理后的数据已保存到: {save_path}")


if __name__ == '__main__':
    #
    data_process(target='NO2(GT)')
    # test_file = '..\\data\\AirQualityUCI_test.xlsx'
    # data_process_test(test_file_path=test_file ,target='NO2(GT)',  save_path="NO2(GT)_test_data.csv")
    # my_train_3()
    # my_test()
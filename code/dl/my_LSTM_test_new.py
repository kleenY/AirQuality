# -*- coding: utf-8 -*
# @Time : 2024/11/23 11:16
# @Author : 杨坤林
# @File : my_test2.py
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
import captum.attr as attr

from tools import *
from tools2 import *
from model import *

import shap


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

def my_test_model(model_save_path, pkl_file_path, scaler_X_path, scaler_y_path, future_hours=1,
                  output_dir="LSTM_results", batch_size=32, target_column='CO(GT)'):
    """
    Function to test the model on the test dataset and perform interpretability analysis using Integrated Gradients.

    Parameters:
    - model_save_path: Path to the saved model
    - pkl_file_path: Path to the pickled test data
    - scaler_X_path: Path to the saved X scaler
    - scaler_y_path: Path to the saved y scaler
    - future_hours: Number of hours to predict
    - output_dir: Directory to save all output files
    - batch_size: Batch size for DataLoader

    Saves:
    - CSV file with evaluation metrics: MAE, RMSE, PMAE, PRMSE
    - Prediction plot for multiple test samples
    - Heatmap showing feature importance based on Integrated Gradients
    """

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载标准化器
    with open(scaler_X_path, 'rb') as file:
        scaler_X = pickle.load(file)
    with open(scaler_y_path, 'rb') as file:
        scaler_y = pickle.load(file)

    # 选择输入特征和目标变量
    selected_features = ['PT08.S5(O3)', 'PT08.S4(NO2)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S1(CO)', target_column]

    # 数据加载与预处理
    with open(pkl_file_path, 'rb') as file:
        data_list = pickle.load(file)

    # 提取输入和输出数据
    test_X = []
    test_y = []

    for sample in data_list:
        input_df, output_df = sample
        input_df = input_df[selected_features]  # 选择特定特征
        test_X.append(input_df.values)
        test_y.append(output_df[target_column])  # 确保输出形状正确

    test_X = np.array(test_X)
    test_y = np.array(test_y)

    # 获取序列长度和预测长度
    seq_len = test_X.shape[1]
    pred_len = test_y.shape[1]
    future_hours = pred_len
    output_size = 1

    # 归一化
    test_X = scaler_X.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    test_y = scaler_y.transform(test_y.reshape(-1, test_y.shape[-1])).reshape(test_y.shape)

    # 转换为 PyTorch 张量
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_y = torch.tensor(test_y.squeeze(), dtype=torch.float32)  # 将 y 的形状简化为 [batch_size, future_hours]

    # 构建测试集的 DataLoader
    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=batch_size)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = my_LSTM(input_size=len(selected_features), output_size=output_size, seq_len=seq_len, pred_len=pred_len,
                    device=device).to(device)

    # 设置 weights_only=True 以避免安全问题
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))

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
    y_pred = y_pred.squeeze()
    y_true = np.concatenate(y_true, axis=0)

    # 反归一化
    y_pred_rescaled = scaler_y.inverse_transform(y_pred.reshape(-1, y_pred.shape[-1])).reshape(y_pred.shape)
    y_true_rescaled = scaler_y.inverse_transform(y_true.reshape(-1, y_true.shape[-1])).reshape(y_true.shape)

    # 计算评估指标
    mae = mean_absolute_error(y_true_rescaled.flatten(), y_pred_rescaled.flatten())
    pmae = mae / np.mean(np.abs(y_pred_rescaled.flatten()))
    rmse = mean_squared_error(y_true_rescaled.flatten(), y_pred_rescaled.flatten(), squared=False)
    prmse = rmse / np.sqrt(np.mean(np.square(y_pred_rescaled.flatten())))

    # 打印结果
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test PMAE: {pmae:.2f}")
    print(f"Test PRMSE: {prmse:.2f}")

    # 保存结果到 CSV 文件
    results = {
        "Metric": ["MAE", "RMSE", "PMAE (%)", "PRMSE (%)"],
        "Value": [mae, rmse, pmae, prmse]
    }
    results_df = pd.DataFrame(results)
    metrics_path = os.path.join(output_dir, "LSTM_test_metrics.csv")
    results_df.to_csv(metrics_path, index=False)
    print(f"Test metrics saved to '{metrics_path}'")

    # 准备保存所有样本的数据
    all_data = []

    # 可视化多个样本的结果
    num_samples_to_visualize = min(10, len(data_list))  # 设置要可视化的样本数量，不超过数据集大小
    time_steps = np.arange(1, future_hours + 1)

    plt.figure(figsize=(15, 10))

    for i in range(num_samples_to_visualize):
        offset = i * (future_hours + 1)  # 添加间距

        # 处理 true_sample 和 pred_sample 是否为标量的情况
        true_sample = y_true_rescaled[i] if isinstance(y_true_rescaled[i], np.ndarray) else [y_true_rescaled[i]]
        pred_sample = y_pred_rescaled[i] if isinstance(y_pred_rescaled[i], np.ndarray) else [y_pred_rescaled[i]]

        plt.plot(np.arange(offset + 1, offset + future_hours + 1), true_sample, label=f"True Sample {i + 1}",
                 marker='o')
        plt.plot(np.arange(offset + 1, offset + future_hours + 1), pred_sample, label=f"Predicted Sample {i + 1}",
                 marker='x')

        # 存储当前样本的数据
        for t in range(future_hours):  # 未来数据的时间步长
            all_data.append({
                'Sample': i + 1,
                'Hour_Ahead': t + 1,
                'True_Value': true_sample[t],
                'Predicted_Value': pred_sample[t]
            })

    # 添加图例和标题
    plt.legend()
    plt.title("Air Quality Prediction (LSTM Test Samples)")
    plt.xlabel("Hours Ahead")
    if target_column == 'CO(GT)':
        plt.ylabel("CO Concentration (Rescaled)")
    elif target_column == 'NOx(GT)':
        plt.ylabel("NOx Concentration (Rescaled)")
    elif target_column == 'NO2(GT)':
        plt.ylabel("NO2 Concentration (Rescaled)")
    plt.xticks(np.arange(1, num_samples_to_visualize * (future_hours + 1), future_hours + 1),
               labels=[f'Sample {i + 1}' for i in range(num_samples_to_visualize)])

    # 保存图像
    plot_path = os.path.join(output_dir, 'LSTM_prediction_test_samples.png')
    plt.savefig(plot_path)
    print(f"Prediction plot saved to '{plot_path}'")

    # 保存所有样本的数据到CSV文件
    data_df = pd.DataFrame(all_data)
    csv_path = os.path.join(output_dir, 'LSTM_prediction_test_samples_data.csv')
    data_df.to_csv(csv_path, index=False)
    print(f"Prediction data saved to '{csv_path}'")

    # 使用 Integrated Gradients 进行可解释性分析
    model.train()

    integrated_gradients = attr.IntegratedGradients(model)
    baseline = torch.zeros_like(test_X[:1])

    # 存储每个时间步的特征重要性
    feature_importances = []

    for step in range(future_hours):  # 遍历每个时间步
        print(f"Calculating attributions for time step {step + 1}/{future_hours}...")

        attributions, delta = integrated_gradients.attribute(
            test_X[:num_samples_to_visualize].to(device),
            baselines=baseline.expand_as(test_X[:num_samples_to_visualize]).to(device),
            n_steps=50,
            internal_batch_size=batch_size,
            target=step,  # 指定时间步索引
            return_convergence_delta=True
        )

        # 取绝对值并求平均以得到每个特征的重要性
        avg_attributions = torch.mean(torch.abs(attributions), dim=(0, 1)).cpu().numpy()
        feature_importances.append(avg_attributions)

        # 绘制每个时间步的特征重要性
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(selected_features)), avg_attributions)
        plt.xticks(range(len(selected_features)), selected_features, rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Average Attribution Magnitude')
        plt.title(f'Feature Importance via Integrated Gradients (Time Step {step + 1})')
        plt.tight_layout()

        # 保存热力图为图像文件
        heatmap_path = os.path.join(output_dir, f'feature_importance_timestep_{step + 1}.png')
        plt.savefig(heatmap_path)
        print(f"Feature importance heatmap for time step {step + 1} saved to '{heatmap_path}'")

    # 汇总所有时间步的重要性
    feature_importances = np.array(feature_importances)  # [future_hours, num_features]

    # 平均时间步的重要性
    mean_importances = np.mean(feature_importances, axis=0)

    # 绘制所有时间步平均特征重要性
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(selected_features)), mean_importances)
    plt.xticks(range(len(selected_features)), selected_features, rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Average Attribution Magnitude (Across All Time Steps)')
    plt.title('Average Feature Importance via Integrated Gradients')
    plt.tight_layout()

    # 保存整体热力图为图像文件
    overall_heatmap_path = os.path.join(output_dir, 'feature_importance_overall.png')
    plt.savefig(overall_heatmap_path)
    print(f"Overall feature importance heatmap saved to '{overall_heatmap_path}'")

    return mae


if __name__ == '__main__':
    # # # 调用示例
    # # # 假设你已经训练完模型并保存了它
    # target_column = 'CO(GT)'
    # pkl_file_path = '..\\data\\split_data\\long_test.pkl'
    # model_save_path = './'+ target_column + '_long_LSTM_output/best_LSTM_model.pth'
    # scaler_X_path = './'+ target_column + '_long_LSTM_output/scaler_X.pkl'
    # scaler_y_path = './'+ target_column + '_long_LSTM_output/scaler_y.pkl'
    # mae = my_test_model(model_save_path=model_save_path, pkl_file_path=pkl_file_path,scaler_X_path = scaler_X_path,
    #                     scaler_y_path=scaler_y_path, target_column=target_column,output_dir=target_column+"_long_LSTM_results")
    # # mae1 = my_test_model_long_term(model_save_path)
    # # mae = my_test_model_long_term_recursive(model_save_path)
    # # mae = my_test_model_long_term_recursive(model_save_path)
    # print(mae)


    # 定义数据类型和目标列
    data_types = ['short', 'medium', 'long']
    target_columns = ['CO(GT)', 'NOx(GT)', 'NO2(GT)']

    # 循环测试
    for data_type in data_types:
        for target_column in target_columns:
            pkl_file_path = f'..\\data\\split_data\\{data_type}_test.pkl'
            model_save_path = f'./{target_column}_{data_type}_LSTM_output/best_LSTM_model.pth'
            scaler_X_path = f'./{target_column}_{data_type}_LSTM_output/scaler_X.pkl'
            scaler_y_path = f'./{target_column}_{data_type}_LSTM_output/scaler_y.pkl'
            output_dir = f'{target_column}_{data_type}_LSTM_results'

            mae = my_test_model(
                model_save_path=model_save_path,
                pkl_file_path=pkl_file_path,
                scaler_X_path=scaler_X_path,
                scaler_y_path=scaler_y_path,
                target_column=target_column,
                output_dir=output_dir
            )
            print(f'MAE for {target_column} ({data_type}): {mae}')
# -*- coding: utf-8 -*
# @Time : 2024/11/12 10:47
# @Author : 杨坤林
# @File : tools.py
# @Software : PyCharm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os




# from tensorflow.keras.optimizers import Adam

data_path = 'G:\\KU\\PhD Course\\Machine Learning\\project\\data\\AirQualityUCI.xlsx'


def plot_air_quality(file_path, columns_to_plot, start_date='2004-03-10'):
    """
    加载空气质量数据，处理缺失值，并绘制指定变量的时间序列图。

    参数:
    - file_path: str, 数据文件路径 (Excel 文件)。
    - columns_to_plot: list, 要绘制的列名称列表。
    - start_date: str, 日期范围的起始日期 (默认值为 '2004-03-10')。
    """
    # 读取 Excel 数据文件
    data = pd.read_excel(file_path)

    # 将 -200 替换为缺失值 NaN，并用列平均值填充
    data[columns_to_plot] = data[columns_to_plot].replace(-200, np.nan)
    data[columns_to_plot] = data[columns_to_plot].apply(lambda x: x.fillna(x.mean()), axis=0)

    # 生成日期范围并添加到 DataFrame
    dates = pd.date_range(start=start_date, periods=len(data), freq='H')
    data['Date'] = dates

    # 计算每列的最小值和最大值
    variation_range = data[columns_to_plot].describe().loc[['min', 'max']]

    # 创建子图
    num_columns = len(columns_to_plot)
    fig, axs = plt.subplots(num_columns, 1, figsize=(14, num_columns * 3), sharex=True)

    # 绘制每个变量的时间序列图
    for i, col in enumerate(columns_to_plot):
        axs[i].plot(data['Date'], data[col])
        axs[i].set_title(col)
        axs[i].set_ylabel(col)

        # 设置 y 轴的范围
        min_val = variation_range.loc['min', col]
        max_val = variation_range.loc['max', col]
        axs[i].set_ylim(min_val, max_val)

    # 设置最后一个子图的 x 轴标签
    axs[-1].set_xlabel('Date')

    plt.tight_layout()
    plt.show()


# file_path = 'AirQualityUCI _ Students.xlsx'
# columns_to_plot = [
#     'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
#     'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
#     'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)'
# ]
# plot_air_quality(file_path, columns_to_plot)



def plot_air_quality_time_series(file_path, columns_to_correct, start_date='2004-03-10'):
    """
    加载空气质量数据，处理缺失值，生成日期列，并绘制指定变量的时间序列图。

    参数:
    - file_path: str, 数据文件路径 (Excel 文件)。
    - columns_to_correct: list, 需要处理和绘制的列名列表。
    - start_date: str, 日期范围的起始日期 (默认值为 '2004-03-10')。
    """
    # 读取 Excel 文件
    data = pd.read_excel(file_path)

    # 将 -200 替换为 NaN
    data[columns_to_correct] = data[columns_to_correct].replace(-200, np.nan)

    # 计算每列的缺失值总数（可选，便于检查）
    missing_values_sum = data[columns_to_correct].isnull().sum()
    print("Missing values per column:\n", missing_values_sum)

    # 使用线性插值法填充缺失值
    data[columns_to_correct] = data[columns_to_correct].interpolate(method='linear')

    # 生成日期范围并添加到 DataFrame
    dates = pd.date_range(start=start_date, periods=len(data), freq='H')
    data['Date'] = dates

    # 计算每列的变动范围
    variation_range = data[columns_to_correct].describe().loc[['min', 'max']]

    # 创建子图
    num_columns = len(columns_to_correct)
    fig, axs = plt.subplots(num_columns, 1, figsize=(14, num_columns * 3), sharex=True)

    # 绘制每个变量的时间序列图
    for i, col in enumerate(columns_to_correct):
        axs[i].plot(data['Date'], data[col])
        axs[i].set_title(col)
        axs[i].set_ylabel(col)

        # 设置 y 轴的范围
        min_val = variation_range.loc['min', col]
        max_val = variation_range.loc['max', col]
        axs[i].set_ylim(min_val, max_val)

    # 设置最后一个子图的 x 轴标签
    axs[-1].set_xlabel('Date')

    plt.tight_layout()
    plt.show()





def plot_air_quality_time_series_with_outlier_removal(file_path, columns_to_correct, start_date='2004-03-10',
                                                      output_folder='plots'):
    """
    加载空气质量数据，处理缺失值，去除离群点，生成日期列，并绘制指定变量的时间序列图，每个子图保存为独立文件。
    保存箱形图和时间序列图，解决字体缺失警告。

    参数:
    - file_path: str, 数据文件路径 (Excel 文件)。
    - columns_to_correct: list, 需要处理和绘制的列名列表。
    - start_date: str, 日期范围的起始日期 (默认值为 '2004-03-10')。
    - output_folder: str, 保存图像的文件夹名称 (默认值为 'plots')。
    """
    # 创建保存图像的文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取 Excel 文件
    data = pd.read_excel(file_path)

    # 将 -200 替换为 NaN
    data[columns_to_correct] = data[columns_to_correct].replace(-200, np.nan)

    # 计算每列的缺失值总数（可选，便于检查）
    missing_values_sum = data[columns_to_correct].isnull().sum()
    print("Missing values per column:\n", missing_values_sum)

    # 使用线性插值法填充缺失值
    data[columns_to_correct] = data[columns_to_correct].interpolate(method='linear')

    # 绘制并保存箱形图
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=data[columns_to_correct])
    plt.title("数值型列的箱形图（离群点检测）", fontname="SimHei")  # 设置支持中文的字体
    boxplot_path = os.path.join(output_folder, "boxplot_outlier_detection.png")
    plt.savefig(boxplot_path)
    plt.show()
    plt.close()
    print(f"Saved boxplot to {boxplot_path}")

    # 绘制并保存直方图
    data[columns_to_correct].hist(bins=20, figsize=(15, 10))
    plt.suptitle("数值型数据分布图", fontname="SimHei")
    hist_path = os.path.join(output_folder, "histogram_distribution.png")
    plt.savefig(hist_path)
    plt.show()
    plt.close()
    print(f"Saved histogram to {hist_path}")

    # 去除离群点 - 使用Z分数法（绝对值大于3的离群点）
    numeric_data = data[columns_to_correct].select_dtypes(include=[np.number])
    z_scores = np.abs(zscore(numeric_data))
    data = data[(z_scores < 3).all(axis=1)]
    print("去除离群点后的数据量:", data.shape)

    # 绘制并保存箱形图(去除离群点之后)
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=data[columns_to_correct])
    plt.title("数值型列的箱形图（去除离群点后）", fontname="SimHei")  # 设置支持中文的字体
    boxplot_path = os.path.join(output_folder, "boxplot_remove_outlier.png")
    plt.savefig(boxplot_path)
    plt.show()
    plt.close()
    print(f"Saved boxplot to {boxplot_path}")

    # 计算并绘制相关矩阵
    correlation_matrix = data[columns_to_correct].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Features', fontname="SimHei")

    # 保存相关矩阵图
    correlation_matrix_path = os.path.join(output_folder, 'correlation_matrix.png')
    plt.savefig(correlation_matrix_path)
    plt.close()
    print(f"Correlation matrix saved to {correlation_matrix_path}")


    # 生成日期范围并添加到 DataFrame
    dates = pd.date_range(start=start_date, periods=len(data), freq='H')
    data['Date'] = dates

    # 计算每列的变动范围
    variation_range = data[columns_to_correct].describe().loc[['min', 'max']]

    # 绘制并保存每个变量的时间序列图
    for col in columns_to_correct:
        plt.figure(figsize=(14, 4))
        plt.plot(data['Date'], data[col])
        plt.title(col, fontname="SimHei")
        plt.ylabel(col, fontname="SimHei")

        # 设置 y 轴的范围
        min_val = variation_range.loc['min', col]
        max_val = variation_range.loc['max', col]
        plt.ylim(min_val, max_val)

        # 设置 x 轴标签
        plt.xlabel('Date')

        # 保存图像
        save_path = os.path.join(output_folder, f"{col}_time_series.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot for {col} to {save_path}")

    print("所有子图已保存到文件夹:", output_folder)

    # return data



def process_air_quality_data(file_path, columns_to_correct, start_date='2004-03-10'):
    """
    加载空气质量数据，处理缺失值，去除离群点，生成日期列，并返回处理后的数据。

    参数:
    - file_path: str, 数据文件路径 (Excel 文件)。
    - columns_to_correct: list, 需要处理的列名列表。
    - start_date: str, 日期范围的起始日期 (默认值为 '2004-03-10')。

    返回:
    - data: 处理后的数据
    """
    # 读取 Excel 文件
    data = pd.read_excel(file_path)

    data = data.dropna(axis=1, how='all')

    # 将 -200 替换为 NaN
    data[columns_to_correct] = data[columns_to_correct].replace(-200, np.nan)

    # 计算每列的缺失值总数（可选，便于检查）
    missing_values_sum = data[columns_to_correct].isnull().sum()
    print("Missing values per column:\n", missing_values_sum)

    # 使用线性插值法填充缺失值
    data[columns_to_correct] = data[columns_to_correct].interpolate(method='linear')

    # # 去除离群点 - 使用Z分数法（绝对值大于3的离群点）
    # numeric_data = data[columns_to_correct].select_dtypes(include=[np.number])
    # z_scores = np.abs(zscore(numeric_data))
    # data = data[(z_scores < 3).all(axis=1)]
    # print("去除离群点后的数据量:", data.shape)

    # 生成日期范围并添加到 DataFrame
    # 1. 合并 Date 和 Time 列
    # 确保数据中包含 Date 和 Time 列
    if 'Date' in data.columns and 'Time' in data.columns:
        # 确保 'Date' 和 'Time' 列为字符串类型
        data['Date'] = data['Date'].astype(str)
        data['Time'] = data['Time'].astype(str)
        # 将 Date 和 Time 合并为一个完整的时间戳
        data['timestamp'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])


    # 2. 检查时间序列是否完整
    # 创建完整的时间序列
    start_time = data['timestamp'].min()
    end_time = data['timestamp'].max()
    complete_time_series = pd.date_range(start=start_time, end=end_time, freq='H')

    # 找出缺失的时间点
    missing_times = complete_time_series.difference(data['timestamp'])

    # 3. 输出检查结果
    if missing_times.empty:
        print("时间序列完整，没有缺失值。")
    else:
        print("时间序列不完整，缺失的时间点如下：")
        print(missing_times)

    # # 4. 如果需要，可以将数据重新对齐补全时间点
    # # 将现有数据与完整时间序列对齐
    # data_complete = pd.DataFrame({'timestamp': complete_time_series})
    # data = pd.merge(data_complete, data, on='timestamp', how='left')

    # # 生成日期范围并添加到 DataFrame
    # dates = pd.date_range(start=start_date, periods=len(data), freq='H')
    # data['Date'] = dates

    # 返回处理后的数据
    return data


# file_path = 'AirQualityUCI _ Students.xlsx'
# columns_to_correct = [
#     'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
#     'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
#     'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
#     'T', 'RH', 'AH'
# ]
# plot_air_quality_time_series(file_path, columns_to_correct)




def preprocess_data(data, columns_to_correct, lag_hours=[1, 2, 3]):
    """
    预处理数据，包括创建目标标签、滞后特征、数据标准化、拆分数据集和相关矩阵的可视化。

    参数:
    - data: 包含原始数据的 pandas DataFrame。
    - columns_to_correct: 需要创建滞后特征的列名列表。
    - lag_hours: 需要创建的滞后小时数列表，默认为 [1, 2, 3]。

    返回:
    - X_train, X_val, X_test: 标准化后的训练、验证、测试特征集。
    - y_train, y_val, y_test: 对应的目标变量。
    """

    # 计算阈值并创建目标标签
    threshold = data['CO(GT)'].mean()
    data['CO_Target'] = (data['CO(GT)'] > threshold).astype(int)

    # 创建滞后特征 (过去1, 2, 3小时)
    for col in columns_to_correct:
        for lag in lag_hours:
            data[f'{col}_lag_{lag}'] = data[col].shift(lag)

    # 去除由滞后特征产生的缺失值
    data.dropna(inplace=True)

    # # 计算并绘制相关矩阵
    # correlation_matrix = data[columns_to_correct].corr()
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    # plt.title('Correlation Matrix of Features')
    # plt.show()

    # 删除不必要的列，构建特征集和目标变量
    columns_to_drop = ["Date", "Time", "CO(GT)", "NMHC(GT)", "PT08.S4(NO2)", "PT08.S3(NOx)", "T", "RH", "AH", "CO_Target"]
    X = data.drop(columns=columns_to_drop)
    y = data['CO_Target']

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 拆分数据集为训练集、验证集和测试集
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

    # 打印拆分后的数据集形状
    print("Shapes of the datasets:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    print("y_test:", y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test

# 使用例子
# X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data, columns_to_correct)




def train_neural_network_pytorch(X_train, y_train, X_val, y_val, neurons_layer1=64, dropout_layer1=0.3, neurons_layer2=32, dropout_layer2=0.3, output_neurons=1, learning_rate=0.0001, epochs=100, batch_size=32):
    """
    This function trains a neural network model for binary classification using PyTorch.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_val: Validation features
    - y_val: Validation labels
    - neurons_layer1: Number of neurons in the first hidden layer
    - dropout_layer1: Dropout rate in the first hidden layer
    - neurons_layer2: Number of neurons in the second hidden layer
    - dropout_layer2: Dropout rate in the second hidden layer
    - output_neurons: Number of neurons in the output layer (for binary classification, should be 1)
    - learning_rate: Learning rate for the optimizer
    - epochs: Number of epochs to train the model
    - batch_size: Batch size for training

    Returns:
    - history: Training history object containing loss and accuracy
    """

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # reshape for binary classification

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    # Create DataLoader for training and validation sets
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    val_data = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Define the neural network model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(X_train.shape[1], neurons_layer1)
            self.dropout1 = nn.Dropout(dropout_layer1)
            self.fc2 = nn.Linear(neurons_layer1, neurons_layer2)
            self.dropout2 = nn.Dropout(dropout_layer2)
            self.fc3 = nn.Linear(neurons_layer2, output_neurons)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout1(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout2(x)
            x = self.fc3(x)
            x = self.sigmoid(x)
            return x

    # Initialize the model
    model = NeuralNetwork()

    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()  # Apply sigmoid threshold of 0.5
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct_train / total_train)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(correct_val / total_val)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), train_losses, label='Training Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), train_accuracies, label='Training Accuracy')
    plt.plot(range(epochs), val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return model, train_losses, val_losses, train_accuracies, val_accuracies



from sklearn.metrics import confusion_matrix, accuracy_score, precision_score


def evaluate_model_performance(model, X_test, y_test, threshold=0.5):
    """
    This function evaluates the model's performance using confusion matrix,
    accuracy, and precision on the test set.

    Parameters:
    - model: The trained model to make predictions
    - X_test: Test feature set
    - y_test: True labels for the test set
    - threshold: The threshold for converting model output to binary class (default is 0.5)

    Returns:
    - accuracy: Accuracy of the model on the test set
    - precision: Precision of the model on the test set
    - cm: Confusion matrix of the model's predictions
    """

    # Predict the test set and convert the output to binary (0 or 1) based on the threshold
    y_pred = (model.predict(X_test) > threshold).astype(int)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Calculate accuracy and precision
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # Print confusion matrix and performance metrics
    print(f"Confusion Matrix:\n{cm}")
    print(f"True Positive (TP): {tp}")
    print(f"True Negative (TN): {tn}")
    print(f"False Positive (FP): {fp}")
    print(f"False Negative (FN): {fn}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")

    return accuracy, precision, cm

# 假设你已经准备好训练好的模型和测试集数据 X_test, y_test
# accuracy, precision, cm = evaluate_model_performance(model, X_test, y_test)


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_air_quality_data(file_path):
    # Load the dataset
    data = pd.read_excel(file_path)

    # Specify the columns to correct
    columns_to_correct = [
        'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
        'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
        'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
        'T', 'RH', 'AH'
    ]

    # Replace -200 with NaN and interpolate missing values
    data[columns_to_correct] = data[columns_to_correct].replace(-200, np.nan)
    data[columns_to_correct] = data[columns_to_correct].interpolate(method='linear')
    data.dropna(inplace=True)

    # Create lagged features (previous 1, 2, 3 hours)
    lag_hours = [1, 2, 3]
    for col in columns_to_correct:
        for lag in lag_hours:
            data[f'{col}_lag_{lag}'] = data[col].shift(lag)

    # Drop rows with NaN values created by lagging
    data = data.dropna()

    # Drop the 'NOx(GT)' column from the features to ensure it is not used in prediction
    X = data.drop(columns=['NOx(GT)', 'Date', 'Time'])
    y = data['NOx(GT)']

    # Convert Date and Time columns to a numerical timestamp
    data['DateTime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
    data['DateTime'] = data['DateTime'].map(pd.Timestamp.timestamp)

    # Adjust the dataset split ratios
    train_size = int(0.7 * len(data))
    val_size = int(0.2 * len(data))
    test_size = len(data) - train_size - val_size

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Print shapes of datasets
    print("X_train shape:", X_train_scaled.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val_scaled.shape)
    print("y_val shape:", y_val.shape)
    print("X_test shape:", X_test_scaled.shape)
    print("y_test shape:", y_test.shape)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test


# 使用方法
# file_path = 'AirQualityUCI _ Students.xlsx'
# X_train, y_train, X_val, y_val, X_test, y_test = preprocess_air_quality_data(file_path)





# 定义神经网络模型类
class RegressionModel(nn.Module):
    def __init__(self, input_dim, a=16, b=0.45, c=8, d=0.45, e=1):
        super(RegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, a),
            nn.ReLU(),
            nn.Dropout(b),
            nn.Linear(a, c),
            nn.ReLU(),
            nn.BatchNorm1d(c),
            nn.Dropout(d),
            nn.Linear(c, e)  # 输出层
        )

    def forward(self, x):
        return self.model(x)


def train_and_evaluate_model(X_train_scaled, y_train, X_val_scaled, y_val,
                             a=16, b=0.45, c=8, d=0.45, e=1, lr=0.001,
                             epochs=100, batch_size=32, save_path='regression_model.pth'):
    """
    使用PyTorch构建、训练、保存并评估回归模型。

    参数:
    - X_train_scaled, y_train: 训练集特征和标签
    - X_val_scaled, y_val: 验证集特征和标签
    - a, b, c, d, e: 模型架构参数 (隐藏层神经元数和Dropout率)
    - lr: 学习率
    - epochs: 训练周期数
    - batch_size: 每批次样本数
    - save_path: 模型保存路径
    """

    # 将数据转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    # 创建数据加载器
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

    # 初始化模型
    input_dim = X_train_scaled.shape[1]
    model = RegressionModel(input_dim, a, b, c, d, e)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 记录训练和验证损失
    train_losses = []
    val_losses = []

    # 训练模型
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        # 计算每个epoch的平均训练损失
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        # 计算每个epoch的平均验证损失
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), save_path)

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 在验证集上进行预测
    model.eval()
    y_val_pred = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            y_pred = model(X_batch)
            y_val_pred.extend(y_pred.numpy())
    y_val_pred = pd.Series(y_val_pred).values.flatten()

    # 绘制实际值与预测值对比图
    plt.figure(figsize=(12, 6))
    plt.plot(y_val.values, label='Actual NOx(GT)')
    plt.plot(y_val_pred, label='Predicted NOx(GT)')
    plt.title('Actual vs Predicted NOx(GT) on Validation Set')
    plt.xlabel('Samples')
    plt.ylabel('NOx(GT)')
    plt.legend()
    plt.show()


# 使用示例
# train_and_evaluate_model(X_train_scaled, y_train, X_val_scaled, y_val)


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def evaluate_regression_model(y_true, y_pred):
    """
    计算并打印回归模型的性能指标：均方根误差 (RMSE) 和平均绝对误差 (MAE)。

    参数:
    - y_true: 测试集的真实标签值
    - y_pred: 测试集的预测标签值

    返回:
    - rmse: 均方根误差 (Root Mean Squared Error)
    - mae: 平均绝对误差 (Mean Absolute Error)
    """
    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # 计算 MAE
    mae = mean_absolute_error(y_true, y_pred)

    # 输出结果
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Number of Samples: {len(y_true)}")

    return rmse, mae


# 使用示例
# rmse, mae = evaluate_regression_model(y_test, y_test_pred)





def load_and_test_pytorch_model(file_path, model_path):
    # Load the dataset
    new_data = pd.read_excel(file_path)

    # Specify the columns to correct
    columns_to_correct = [
        'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
        'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
        'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
        'T', 'RH', 'AH'
    ]

    # Replace -200 with NaN
    new_data[columns_to_correct] = new_data[columns_to_correct].replace(-200, np.nan)

    # Interpolate missing values
    new_data[columns_to_correct] = new_data[columns_to_correct].interpolate(method='linear')

    # Drop rows with any remaining NaN values
    new_data.dropna(inplace=True)

    # Create lagged features (previous 1, 2, 3 hours)
    lag_hours = [1, 2, 3]
    for col in columns_to_correct:
        for lag in lag_hours:
            new_data[f'{col}_lag_{lag}'] = new_data[col].shift(lag)

    # Drop rows with NaN values created by lagging
    new_data = new_data.dropna()

    # Drop the 'NOx(GT)' column from the features to ensure it is not used in prediction
    X_new = new_data.drop(columns=['NOx(GT)', 'Date', 'Time'])
    y_new = new_data['NOx(GT)']

    # Convert the Date and Time columns to datetime and then to numerical features
    new_data['DateTime'] = pd.to_datetime(new_data['Date'].astype(str) + ' ' + new_data['Time'].astype(str))
    new_data['DateTime'] = new_data['DateTime'].map(pd.Timestamp.timestamp)

    # Scale the data
    scaler = StandardScaler()
    X_new_scaled = scaler.fit_transform(X_new)

    # Convert scaled features to PyTorch tensor
    X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

    # Load the saved PyTorch model
    model = torch.load(model_path)
    model.eval()  # Set model to evaluation mode

    # Make predictions on the new data
    with torch.no_grad():
        predictions = model(X_new_tensor).numpy().flatten()

    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_new, predictions))
    mae = mean_absolute_error(y_new, predictions)

    # Print performance metrics
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Number of Samples: {len(y_new)}")

    # Plot true values vs predictions
    plt.figure(figsize=(10, 5))
    plt.plot(y_new.values, label='True Values')
    plt.plot(predictions, label='Predictions')
    plt.legend()
    plt.title('True Values vs Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('NOx(GT)')
    plt.show()

    # Return predictions and true values for further analysis
    return pd.DataFrame({'True Values': y_new, 'Predictions': predictions})


# # Usage example
# file_path = 'G:/dataset/Generalization Dataset.xlsx'
# model_path = 'G:/dataset/regression_model.pth'  # Change to your PyTorch model path
# results = load_and_test_pytorch_model(file_path, model_path)



# Function to load the saved model and test new data (PyTorch version)
def load_and_test_classification_model_pytorch(file_path, model_path, threshold):
    # Load the dataset
    new_data = pd.read_excel(file_path)

    # Specify the columns to correct (if necessary)
    columns_to_correct = [
        'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
        'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
        'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
        'T', 'RH', 'AH'
    ]

    # Replace -200 with NaN (if necessary)
    new_data[columns_to_correct] = new_data[columns_to_correct].replace(-200, np.nan)

    # Interpolate missing values
    new_data[columns_to_correct] = new_data[columns_to_correct].interpolate(method='linear')

    # Drop rows with any remaining NaN values
    new_data.dropna(inplace=True)

    # Create lagged features (previous 1, 2, 3 hours)
    lag_hours = [1, 2, 3]
    for col in columns_to_correct:
        for lag in lag_hours:
            new_data[f'{col}_lag_{lag}'] = new_data[col].shift(lag)

    # Drop rows with NaN values created by lagging
    new_data = new_data.dropna()

    # Create the target labels
    new_data['CO_Target'] = (new_data['CO(GT)'] > threshold).astype(int)

    # Split the features and target
    X_new = new_data.drop(columns=['CO_Target', 'Date', 'Time', 'CO(GT)'])
    y_new = new_data['CO_Target']

    # Convert the Date and Time columns to datetime and then to numerical features
    new_data['DateTime'] = pd.to_datetime(new_data['Date'].astype(str) + ' ' + new_data['Time'].astype(str))
    new_data['DateTime'] = new_data['DateTime'].map(pd.Timestamp.timestamp)

    # Scale the data
    scaler = StandardScaler()
    X_new_scaled = scaler.fit_transform(X_new)

    # Convert data to PyTorch tensors
    X_new_scaled_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)
    y_new_tensor = torch.tensor(y_new.values, dtype=torch.long)  # PyTorch expects long for classification labels

    # Load the saved model (PyTorch model)
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode

    # Make predictions on the new data
    with torch.no_grad():  # No need to calculate gradients during inference
        outputs = model(X_new_scaled_tensor)
        predictions = torch.sigmoid(outputs).squeeze().numpy()  # Apply sigmoid for binary classification
        predicted_classes = (predictions > 0.5).astype(int)

    # Calculate performance metrics
    accuracy = accuracy_score(y_new, predicted_classes)
    precision = precision_score(y_new, predicted_classes, average='weighted')
    recall = recall_score(y_new, predicted_classes, average='weighted')
    f1 = f1_score(y_new, predicted_classes, average='weighted')
    cm = confusion_matrix(y_new, predicted_classes)

    # Print performance metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Number of Samples: {len(y_new)}")

    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.show()

    # Return metrics and results
    return accuracy, precision, recall, f1, cm, predictions, predicted_classes


# file_path = 'G:/dataset/Generalization Dataset.xlsx'
# model_path = 'G:/dataset/classification_model.pth'  # 你的PyTorch模型路径
# threshold = 100  # 选择合适的阈值
# accuracy, precision, recall, f1, cm, predictions, predicted_classes = load_and_test_classification_model_pytorch(file_path, model_path, threshold)

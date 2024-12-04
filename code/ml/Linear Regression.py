import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import torch
import os
import matplotlib.pyplot as plt
import shap
import pandas as pd

# 定义提取特征和标签的函数
def extract_features_and_labels(dataset, feature_columns, label_column):
    X = []
    y = []
    for input_data, label_data in dataset:
        X.append(input_data[feature_columns].values)
        y.append(label_data[label_column].values)
    X = np.array(X)
    y = np.array(y)
    return X, y


# 获取短期、中期和长期模式的训练集和测试集
split_modes = {
    '短期': (4, 1),   # 短期模式：输入4小时，预测1小时
    '中期': (12, 3),    # 中期模式：输入12小时，预测3小时
    '长期': (24, 6)    # 长期模式：输入24小时，预测4小时
}


# 处理每种模式的数据集
labels = ['NOx(GT)', 'NO2(GT)', 'CO(GT)']

for label in labels:
    for mode, (input_size, predict_size) in split_modes.items():
        # 读取相应模式的训练和测试集
        train_file_path = f'./sp_data/split_data/{mode}_train.pkl'
        test_file_path = f'./sp_data/split_data/{mode}_test.pkl'

        with open(train_file_path, 'rb') as f:
            train_data = pickle.load(f)

        with open(test_file_path, 'rb') as f:
            test_data = pickle.load(f)

        #print(f"读取 {mode} 模式的训练集和测试集：")
        #print(f"训练集长度：{len(train_data)}")
        #print(f"测试集长度：{len(test_data)}")

        # 提取特征列【CO(GT)】,或【NO2(GT)】，或【NOx(GT)】
        features = [str(label), 'PT08.S5(O3)', 'PT08.S4(NO2)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S1(CO)']

        # 提取训练集和测试集的特征和标签
        X_train, y_train = extract_features_and_labels(train_data, features, label)
        X_test, y_test = extract_features_and_labels(test_data, features, label)
        
        # 打印形状以进行调试
        #print(f"X_train shape: {X_train.shape}")
        #print(f"y_train shape: {y_train.shape}")
        #print(f"X_test shape: {X_test.shape}")
        #print(f"y_test shape: {y_test.shape}")

        # 将数据转换为 PyTorch 张量
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.squeeze(), dtype=torch.float32)  # 将 y 的形状简化为 [batch_size, future_hours]
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.squeeze(), dtype=torch.float32)

        #print(f"拼接后的 {mode} 训练集和测试集的形状：")
        #print(f"X_train: {X_train_tensor.shape}, y_train: {y_train_tensor.shape}")
        #print(f"X_test: {X_test_tensor.shape}, y_test: {y_test_tensor.shape}")

        X_train_flat = X_train_tensor.reshape(X_train_tensor.shape[0], -1)
        X_test_flat = X_test_tensor.reshape(X_test_tensor.shape[0], -1)

        # 打印转换后的形状以进行调试
        # print(f"X_train_flat shape: {X_train_flat.shape}")
        # print(f"X_test_flat shape: {X_test_flat.shape}")

        # 训练 Linear Regression 模型
        model = LinearRegression()
        model.fit(X_train_flat.numpy(), y_train_tensor.numpy())

        # 使用测试集进行预测
        y_pred = model.predict(X_test_flat.numpy())

        # 计算模型评价指标
        mae = mean_absolute_error(y_test_tensor.numpy(), y_pred)
        pmae = mae / np.mean(np.abs(y_test_tensor.numpy()))  # PMAE = MAE / 平均真实值
        rmse = np.sqrt(mean_squared_error(y_test_tensor.numpy(), y_pred))
        prmse = rmse / np.sqrt(np.mean(np.square(y_test_tensor.numpy())))  # PRMSE = RMSE

        # 打印评价指标
        print(f"模型针对 {label} 的{mode} 预测：")
        print(f"MAE: {mae}")
        print(f"PMAE: {pmae}")
        print(f"RMSE: {rmse}")
        print(f"PRMSE: {prmse}")
        

        
        # 创建保存模型的目录
        model_dir = "./model"
        os.makedirs(model_dir, exist_ok=True)

        # 保存评价指标
        """metrics_filename = f"./model/linear_regression_{mode}_{label}_metrics.txt"
        with open(metrics_filename, 'w') as f:
            f.write(f"MAE: {mae}\n")
            f.write(f"PMAE: {pmae}\n")
            f.write(f"RMSE: {rmse}\n")
            f.write(f"PRMSE: {prmse}\n")"""


        # 保存拼接后训练的模型
        # model_filename = f"./model/linear_regression_{mode}_{label}_model_combined.pkl"
        # joblib.dump(model, model_filename)

        
        # 提取预测的目标变量部分
        # print(f"y_pred shape: {y_pred.shape}")
        # print(f"y_test shape: {y_test_tensor.shape}")

        y_pred_target = y_pred
        y_test_target = y_test_tensor.numpy()
        print(f"y_pred_target shape: {y_pred_target.shape}")
        print(f"y_test_target shape: {y_test_target.shape}")
        
        y_pred_target2 = y_pred.flatten()
        y_test_target2 = y_test_target.flatten()
        print(f"y_pred_target2 shape: {y_pred_target2.shape}")
        print(f"y_test_target2 shape: {y_test_target2.shape}")



        if mode == '短期':
            Flag = 'Short-term'
        elif mode == '中期':
            Flag = 'Medium-term'
        elif mode == '长期':
            Flag = 'Long-term'

        # 记录画图数据
        data = []
        num_samples_to_plot = min(10, len(y_test_target)) 
        for i in range(num_samples_to_plot):
            start_idx = i * predict_size
            end_idx = start_idx + predict_size
            for j in range(predict_size):
                data.append([i + 1, j + 1, y_test_target2[start_idx + j], y_pred_target2[start_idx + j]])
                print(f"Sample {i + 1}, Hour Ahead {j + 1}, True Value: {y_test_target2[start_idx + j]}, Predicted Value: {y_pred_target2[start_idx + j]}")
        
        df = pd.DataFrame(data, columns=['Sample', 'Hour_Ahead', 'True_Value', 'Predicted_Value'])
        csv_filename = f"{model_dir}/linear_regression_{Flag}_{label}_prediction_data.csv"
        df.to_csv(csv_filename, index=False)

        # 绘制预测值与真实值的对比图
        plt.figure(figsize=(15, 8))
        # num_samples_to_plot = min(10, len(y_test_target)) 

        for i in range(num_samples_to_plot):
            start_idx = i * predict_size
            end_idx = start_idx + predict_size
            plt.plot(range(start_idx, end_idx), y_test_target[i], label=f'True Sample {i + 1}', marker='o', color=f'C{i}')
            plt.plot(range(start_idx, end_idx), y_pred_target[i], label=f'Predicted Sample {i + 1}', linestyle='--', marker='x', color=f'C{i}')

        plt.xlabel('Hours Ahead')
        plt.ylabel(f'{label}')
        plt.title(f'Air Quality Prediction - Linear Regression  - {Flag}')
        plt.legend()
        plt.grid(True)
        plot_filename = f"{model_dir}/linear_regression_{Flag}_{label}_prediction_plot.png"
        # plt.savefig(plot_filename)
        # plt.show()
        plt.close()
            


        """# 使用 SHAP 解释器
        X_train_flat_np = X_train_flat.numpy()
        X_test_flat_np = X_test_flat.numpy()
        masker = shap.maskers.Independent(X_train_flat_np)
        explainer = shap.Explainer(model, masker)
        shap_values = explainer(X_test_flat_np)
        
        # 绘制 SHAP 值图
        extended_features = [f"{feature}_t-{j}" for feature in features for j in range(input_size)]
        sort_inds = np.argsort(np.sum(np.abs(shap_values.values), axis=0))
        shap.summary_plot(shap_values, X_test_flat_np, feature_names=np.array(extended_features)[sort_inds], plot_type="bar",show=False)
        # 保存 SHAP 值图
        shap_plot_filename = f"{model_dir}/linear_regression_{Flag}_{label}_shap_plot.png"
        plt.savefig(shap_plot_filename)
        plt.close()"""

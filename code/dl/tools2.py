# -*- coding: utf-8 -*
# @Time : 2024/11/11 23:52
# @Author : 杨坤林
# @File : tools2.py
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

from joblib import dump, load


def preprocess_data(file_path, fill_na_with_median=False):
    # 加载数据
    data = pd.read_csv(file_path)

    # 查看数据基本信息
    print("数据基本信息：")
    print(data.info())
    print("\n前5行数据：")
    print(data.head())
    print("\n数据统计描述：")
    print(data.describe())

    # 直方图 - 查看数值型数据的分布
    data.hist(bins=20, figsize=(15, 10))
    plt.suptitle("数值型数据分布图")
    plt.show()

    # 时间序列图 - 假设有一个名为 'timestamp' 的时间列
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])  # 确保时间列为datetime格式
        data.set_index('timestamp').plot(figsize=(15, 6))
        plt.title("时间序列图")
        plt.show()

    # 检查缺失值并处理
    print("\n缺失值概览：")
    print(data.isnull().sum())
    if fill_na_with_median:
        data.fillna(data.median(), inplace=True)  # 用中位数填补缺失值
    else:
        data.dropna(inplace=True)  # 删除缺失值行

    # 使用箱形图查看离群值
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=data.select_dtypes(include=[np.number]))
    plt.title("数值型列的箱形图（离群点检测）")
    plt.show()

    # 去除离群点 - 使用Z分数法（绝对值大于3的离群点）
    data = data[(np.abs(zscore(data.select_dtypes(include=[np.number]))) < 3).all(axis=1)]
    print("去除离群点后的数据量:", data.shape)

    # 确保时间列格式一致
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        print("时间列转换完成，格式为：", data['timestamp'].dtype)

    # 数值标准化（StandardScaler）或归一化（MinMaxScaler）
    scaler = StandardScaler()  # 使用Z-score标准化
    # scaler = MinMaxScaler()  # 或者使用Min-Max归一化
    numerical_features = data.select_dtypes(include=[np.number]).columns
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    print("标准化/归一化处理完成")

    # 编码分类变量
    categorical_features = data.select_dtypes(include=[object]).columns
    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    print("分类变量编码完成")

    # 查看预处理后的数据
    print("\n预处理完成的数据集：")
    print(data.head())

    return data


# # 假设数据文件路径为 '/path/to/your/data.csv'
# file_path = '/path/to/your/data.csv'
# preprocessed_data = preprocess_data(file_path, fill_na_with_median=False)

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def feature_extraction(data):
    """
    提取时间特征和地理信息（如有）
    """

    # # 创建保存目录（如果不存在）
    # os.makedirs(save_folder, exist_ok=True)

    # 确保数据中包含 Date 和 Time 列
    if 'timestamp' in data.columns:

        # # 确保 'Date' 和 'Time' 列为字符串类型
        # data['Date'] = data['Date'].astype(str)
        # data['Time'] = data['Time'].astype(str)
        # # 将 Date 和 Time 合并为一个完整的时间戳
        # data['timestamp'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])

        # 提取时间特征
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        data['season'] = data['timestamp'].dt.month % 12 // 3 + 1

        # # 可视化时间特征的分布
        # time_features = ['hour', 'day_of_week', 'month', 'season']
        # for feature in time_features:
        #     fig = plt.figure(figsize=(8, 6))
        #     sns.countplot(data=data, x=feature, palette='viridis')
        #     plt.title(f'Distribution of {feature}')
        #     plt.tight_layout()
        #     # 保存图片
        #     fig.savefig(os.path.join(save_folder, f'{feature}_distribution.png'))
        #     plt.close(fig)  # 防止内存占用

    else:
        print("数据中未找到 'Date' 和 'Time' 列。无法提取时间特征。")

    # # 假设有位置或土地利用数据（替换 'location' 和 'land_use'）
    # if 'location' in data.columns:
    #     fig = plt.figure(figsize=(8, 6))
    #     sns.countplot(data=data, x='location', palette='viridis')
    #     plt.title('Location Feature Distribution')
    #     plt.tight_layout()
    #     fig.savefig(os.path.join(save_folder, 'location_feature_distribution.png'))
    #     plt.close(fig)
    #
    # if 'land_use' in data.columns:
    #     fig = plt.figure(figsize=(8, 6))
    #     sns.countplot(data=data, x='land_use', palette='viridis')
    #     plt.title('Land Use Feature Distribution')
    #     plt.tight_layout()
    #     fig.savefig(os.path.join(save_folder, 'land_use_feature_distribution.png'))
    #     plt.close(fig)

    print("特征提取完成")
    return data


def feature_selection(data, target_column, n_components=5):
    """
    选择相关特征和降维
    """
    # 计算相关性并筛选与目标变量高度相关的特征
    correlation_matrix = data.corr()
    target_correlation = correlation_matrix[target_column].abs().sort_values(ascending=False)
    selected_features = target_correlation[target_correlation > 0.1].index  # 示例阈值0.1
    data = data[selected_features]

    # 使用PCA进行降维
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data.drop(columns=[target_column]))
    print("PCA降维完成，保留{}个主成分".format(n_components))

    return pd.DataFrame(data_pca, columns=[f'PC{i + 1}' for i in range(n_components)]), data[target_column]






# # 假设目标列名为 'target'
# data = feature_extraction(data)
# data_pca, target = feature_selection(data, target_column='target', n_components=5)
# data_with_interactions = create_interaction_features(data)



from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures



# 特征工程展示

import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures


def feature_extraction_show(data, save_folder="feature_visualizations"):
    """
    提取时间特征和地理信息（如有）
    """

    # 创建保存目录（如果不存在）
    os.makedirs(save_folder, exist_ok=True)

    # 确保数据中包含 Date 和 Time 列
    if 'timestamp' in data.columns:

        # # 确保 'Date' 和 'Time' 列为字符串类型
        # data['Date'] = data['Date'].astype(str)
        # data['Time'] = data['Time'].astype(str)
        # # 将 Date 和 Time 合并为一个完整的时间戳
        # data['timestamp'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])

        # 提取时间特征
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        data['season'] = data['timestamp'].dt.month % 12 // 3 + 1

        # 可视化时间特征的分布
        time_features = ['hour', 'day_of_week', 'month', 'season']
        for feature in time_features:
            fig = plt.figure(figsize=(8, 6))
            sns.countplot(data=data, x=feature, palette='viridis')
            plt.title(f'Distribution of {feature}')
            plt.tight_layout()
            # 保存图片
            fig.savefig(os.path.join(save_folder, f'{feature}_distribution.png'))
            plt.close(fig)  # 防止内存占用

    else:
        print("数据中未找到 'Date' 和 'Time' 列。无法提取时间特征。")

    # # 假设有位置或土地利用数据（替换 'location' 和 'land_use'）
    # if 'location' in data.columns:
    #     fig = plt.figure(figsize=(8, 6))
    #     sns.countplot(data=data, x='location', palette='viridis')
    #     plt.title('Location Feature Distribution')
    #     plt.tight_layout()
    #     fig.savefig(os.path.join(save_folder, 'location_feature_distribution.png'))
    #     plt.close(fig)
    #
    # if 'land_use' in data.columns:
    #     fig = plt.figure(figsize=(8, 6))
    #     sns.countplot(data=data, x='land_use', palette='viridis')
    #     plt.title('Land Use Feature Distribution')
    #     plt.tight_layout()
    #     fig.savefig(os.path.join(save_folder, 'land_use_feature_distribution.png'))
    #     plt.close(fig)

    print("特征提取完成")
    return data


def save_figure(fig, folder_path, filename):
    """保存图形到指定文件夹"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # 创建文件夹
    fig.savefig(os.path.join(folder_path, filename))
    plt.close(fig)  # 关闭图形以释放内存


def feature_selection_show(data, target_column, variance_threshold=0.95, save_folder="pca_visualizations"):
    """
    执行PCA降维并可视化结果，保留指定方差比例的主成分。

    参数：
    - data: pd.DataFrame，包含原始数据。
    - target_column: str，目标列的名称。
    - variance_threshold: float，PCA要保留的方差比例（例如95%）。
    - save_folder: str，保存图像的文件夹路径。

    返回：
    - data_pca: pd.DataFrame，包含主成分的数据框。
    """
    # 检查并创建保存文件夹（如果不存在）
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # # 计算相关性并筛选与目标变量高度相关的特征
    # correlation_matrix = data.corr()
    # target_correlation = correlation_matrix[target_column].abs().sort_values(ascending=False)
    # selected_features = target_correlation[target_correlation > 0.1].index  # 示例阈值0.1
    # data = data[selected_features]

    # 使用PCA进行降维，保留足够的主成分直到累计解释的方差达到variance_threshold
    pca = PCA()
    data_pca = pca.fit_transform(data.drop(columns=[target_column, 'Date', 'Time', 'timestamp']))
    cumulative_variance = pca.explained_variance_ratio_.cumsum()

    # 找到保留的主成分数量
    n_components = (cumulative_variance <= variance_threshold).sum() + 1
    print(f"PCA降维完成，保留{n_components}个主成分，累计方差解释比例达到{cumulative_variance[n_components-1]:.2f}")

    # 重新进行PCA，保留需要的主成分数量
    pca = PCA(n_components=n_components)
    # 保存PCA模型
    data_pca = pca.fit_transform(data.drop(columns=[target_column, 'Date', 'Time', 'timestamp']))
    dump(pca, target_column + '_pca_model.joblib')

    # 获取PCA组件的权重（原始特征对每个主成分的贡献度）
    pca_components = pca.components_
    feature_names = data.drop(columns=[target_column, 'Date', 'Time', 'timestamp']).columns

    # 绘制方差解释比例
    fig = plt.figure(figsize=(8, 6))
    plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_, alpha=0.7, color='b', label='Explained Variance Ratio')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'PCA - Explained Variance Ratio for {n_components} Components')
    plt.xticks(range(1, n_components + 1), [f'PC{i+1}' for i in range(n_components)], rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/PCA_Explained_Variance_Ratio.png")
    # plt.show()
    plt.close()

    # 如果降维后有2个主成分，绘制2D散点图
    if n_components >= 2:
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5, color='r')
        plt.xlabel(f'Principal Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'Principal Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2f})')
        plt.title(f'PCA - 2D Scatter Plot (Top 2 Components)')
        plt.tight_layout()
        plt.savefig(f"{save_folder}/PCA_2D_Scatter_Plot.png")
        # plt.show()
        plt.close()

    # 画出每个主成分的特征贡献度（即每个特征在主成分中的权重）
    for i in range(n_components):
        fig, ax = plt.subplots(figsize=(8, 6))
        feature_contributions = pd.Series(pca_components[i], index=feature_names).sort_values(ascending=False)
        sns.barplot(x=feature_contributions.index, y=feature_contributions.values, ax=ax, palette="viridis")
        ax.set_xlabel('Feature')
        ax.set_ylabel('Contribution to Principal Component')
        ax.set_title(f'Feature Contribution to PC{i+1} for ' + target_column)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"{save_folder}/Feature_Contribution_to_PC{i+1}.png")
        # plt.show()
        plt.close()

    return pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(n_components)]), data[target_column]



def feature_selection_test(data, target_column, variance_threshold=0.95, save_folder="pca_test", load_path=''):
    """
    执行PCA降维并可视化结果，保留指定方差比例的主成分。

    参数：
    - data: pd.DataFrame，包含原始数据。
    - target_column: str，目标列的名称。
    - variance_threshold: float，PCA要保留的方差比例（例如95%）。
    - save_folder: str，保存图像的文件夹路径。

    返回：
    - data_pca: pd.DataFrame，包含主成分的数据框。
    """
    # 检查并创建保存文件夹（如果不存在）
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # # 计算相关性并筛选与目标变量高度相关的特征
    # correlation_matrix = data.corr()
    # target_correlation = correlation_matrix[target_column].abs().sort_values(ascending=False)
    # selected_features = target_correlation[target_correlation > 0.1].index  # 示例阈值0.1
    # data = data[selected_features]
    #
    # # 使用PCA进行降维，保留足够的主成分直到累计解释的方差达到variance_threshold
    # pca = PCA()
    # data_pca = pca.fit_transform(data.drop(columns=[target_column]))
    # cumulative_variance = pca.explained_variance_ratio_.cumsum()

    # # 找到保留的主成分数量
    # n_components = (cumulative_variance <= variance_threshold).sum() + 1
    # print(f"PCA降维完成，保留{n_components}个主成分，累计方差解释比例达到{cumulative_variance[n_components-1]:.2f}")

    # 重新进行PCA，保留需要的主成分数量
    # 加载PCA模型，并对测试集进行转换
    pca = load(load_path)
    # pca = PCA(n_components=n_components)
    # 保存PCA模型
    # dump(pca, target_column + '_pca_model.joblib')
    data_pca = pca.fit_transform(data.drop(columns=[target_column, 'Date', 'Time', 'timestamp']))

    # # 获取PCA组件的权重（原始特征对每个主成分的贡献度）
    # pca_components = pca.components_
    # feature_names = data.drop(columns=[target_column]).columns

    # # 绘制方差解释比例
    # fig = plt.figure(figsize=(8, 6))
    # plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_, alpha=0.7, color='b', label='Explained Variance Ratio')
    # plt.xlabel('Principal Component')
    # plt.ylabel('Explained Variance Ratio')
    # plt.title(f'PCA - Explained Variance Ratio for {n_components} Components')
    # plt.xticks(range(1, n_components + 1), [f'PC{i+1}' for i in range(n_components)], rotation=45)
    # plt.tight_layout()
    # plt.savefig(f"{save_folder}/PCA_Explained_Variance_Ratio.png")
    # # plt.show()
    # plt.close()
    #
    # # 如果降维后有2个主成分，绘制2D散点图
    # if n_components >= 2:
    #     fig = plt.figure(figsize=(8, 6))
    #     plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5, color='r')
    #     plt.xlabel(f'Principal Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2f})')
    #     plt.ylabel(f'Principal Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2f})')
    #     plt.title(f'PCA - 2D Scatter Plot (Top 2 Components)')
    #     plt.tight_layout()
    #     plt.savefig(f"{save_folder}/PCA_2D_Scatter_Plot.png")
    #     # plt.show()
    #     plt.close()
    #
    # # 画出每个主成分的特征贡献度（即每个特征在主成分中的权重）
    # for i in range(n_components):
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     feature_contributions = pd.Series(pca_components[i], index=feature_names).sort_values(ascending=False)
    #     sns.barplot(x=feature_contributions.index, y=feature_contributions.values, ax=ax, palette="viridis")
    #     ax.set_xlabel('Feature')
    #     ax.set_ylabel('Contribution to Principal Component')
    #     ax.set_title(f'Feature Contribution to PC{i+1}')
    #     plt.xticks(rotation=90)
    #     plt.tight_layout()
    #     plt.savefig(f"{save_folder}/Feature_Contribution_to_PC{i+1}.png")
    #     # plt.show()
    #     plt.close()
    if target_column == 'CO(GT)':
        n_components = 4
    elif target_column == 'NOx(GT)':
        n_components = 3
    elif target_column == 'NO2(GT)':
        n_components = 4

    return pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(n_components)]), data[target_column]

def feature_selection_show_2(data, target_column, variance_threshold=0.95, save_folder="pca_visualizations"):
    """
    执行PCA降维并可视化结果，保留指定方差比例的主成分。

    参数：
    - data: pd.DataFrame，包含原始数据。
    - target_column: str，目标列的名称。
    - variance_threshold: float，PCA要保留的方差比例（例如95%）。
    - save_folder: str，保存图像的文件夹路径。

    返回：
    - data_pca: pd.DataFrame，包含主成分的数据框。
    """
    # 检查并创建保存文件夹（如果不存在）
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 计算相关性并筛选与目标变量高度相关的特征
    correlation_matrix = data.corr()
    target_correlation = correlation_matrix[target_column].abs().sort_values(ascending=False)
    selected_features = target_correlation[target_correlation > 0.1].index  # 示例阈值0.1
    data = data[selected_features]

    # 使用PCA进行降维，保留足够的主成分直到累计解释的方差达到variance_threshold
    pca = PCA()
    data_pca = pca.fit_transform(data)  # 这里去掉了target_column的处理，因为目标历史特征也参与降维
    cumulative_variance = pca.explained_variance_ratio_.cumsum()

    # 找到保留的主成分数量，直到累计方差达到指定的阈值
    n_components = (cumulative_variance <= variance_threshold).sum() + 1
    print(f"PCA降维完成，保留{n_components}个主成分，累计方差解释比例达到{cumulative_variance[n_components-1]:.2f}")

    # 重新进行PCA，保留需要的主成分数量
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)

    # 获取PCA组件的权重（原始特征对每个主成分的贡献度）
    pca_components = pca.components_
    feature_names = data.columns

    # 绘制方差解释比例
    fig = plt.figure(figsize=(8, 6))
    plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_, alpha=0.7, color='b', label='Explained Variance Ratio')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'PCA - Explained Variance Ratio for {n_components} Components')
    plt.xticks(range(1, n_components + 1), [f'PC{i+1}' for i in range(n_components)], rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/PCA_Explained_Variance_Ratio.png")
    # plt.show()
    plt.close()

    # 如果降维后有2个主成分，绘制2D散点图
    if n_components >= 2:
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5, color='r')
        plt.xlabel(f'Principal Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'Principal Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2f})')
        plt.title(f'PCA - 2D Scatter Plot (Top 2 Components)')
        plt.tight_layout()
        plt.savefig(f"{save_folder}/PCA_2D_Scatter_Plot.png")
        # plt.show()
        plt.close()

    # 画出每个主成分的特征贡献度（即每个特征在主成分中的权重）
    for i in range(n_components):
        fig, ax = plt.subplots(figsize=(8, 6))
        feature_contributions = pd.Series(pca_components[i], index=feature_names).sort_values(ascending=False)
        sns.barplot(x=feature_contributions.index, y=feature_contributions.values, ax=ax, palette="viridis")
        ax.set_xlabel('Feature')
        ax.set_ylabel('Contribution to Principal Component')
        ax.set_title(f'Feature Contribution to PC{i+1}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"{save_folder}/Feature_Contribution_to_PC{i+1}.png")
        # plt.show()
        plt.close()

    return pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(n_components)]), data[target_column]



from sklearn.preprocessing import PolynomialFeatures


def create_interaction_features_from_pca(pca_data, target_column, data, poly_degree=2, threshold=0.1,
                                         save_folder="pca_visualizations", top_n=6):
    """
    基于PCA降维后的数据，创建交互特征或多项式特征，限制交互特征数量。
    """
    # 1. 使用PolynomialFeatures生成交互特征或多项式特征
    poly = PolynomialFeatures(degree=poly_degree, interaction_only=False, include_bias=False)
    interaction_features = poly.fit_transform(pca_data)

    # 2. 通过PCA数据的列名来命名交互特征
    feature_names = poly.get_feature_names_out([f'PC{i+1}' for i in range(pca_data.shape[1])])

    # 3. 将交互特征转换为DataFrame，并给它们命名
    interaction_df = pd.DataFrame(interaction_features, columns=feature_names)

    # # 4. 将生成的交互特征与原数据合并，只用PCA的特征，不包含原始数据特征
    # pca_column_names = [f'PC{i+1}' for i in range(pca_data.shape[1])]
    # data_with_interactions = pd.DataFrame(pca_data, columns=pca_column_names)

    # 5. 修改交互特征列名，确保与PCA特征不重名
    # interaction_df.columns = [f'inter_{col}' for col in interaction_df.columns]

    # 6. 将交互特征加入到PCA数据框中
    data_with_interactions = pd.concat([interaction_df, data[target_column]], axis=1)

    # 7. 计算新生成特征与目标变量的相关性
    correlation_matrix_new = data_with_interactions.corr()
    target_correlation_new = correlation_matrix_new[target_column].abs().sort_values(ascending=False)

    # 8. 筛选出与目标变量相关性较高的交互特征（根据阈值）
    selected_features = target_correlation_new[target_correlation_new > threshold].index

    # 9. 限制保留top_n个与目标变量相关性最强的特征（包括PCA特征和交互特征）
    # top_n_features = selected_features[:top_n]

    # 获取保留的特征数据
    data_selected = data_with_interactions[selected_features]

    # 可视化相关性（PCA特征和新生成的交互特征与目标列的相关性）
    plt.figure(figsize=(12, 6))
    sns.barplot(x=target_correlation_new.index, y=target_correlation_new.values)
    plt.title(f"Correlation of New Interaction Features with {target_column}")
    plt.xlabel("Features")
    plt.ylabel("Correlation with Target")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_folder}/PCA_and_Inter_Feature_Correlation_with_Target.png")
    # plt.show()
    plt.close()


    return data_selected, data_with_interactions[target_column]












# ARIMA 模型函数（不变，仍使用 statsmodels 库）
def arima_model(data, order=(1, 1, 1)):
    """
    ARIMA 模型的构建与拟合
    """
    model = ARIMA(data, order=order)
    arima_result = model.fit()
    print("ARIMA 模型训练完成")
    return arima_result





def create_lstm_dataset(data, n_steps):
    """
    创建 LSTM 数据集窗口
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # 增加一个维度以匹配 LSTM 输入格式
    y = torch.tensor(y, dtype=torch.float32)
    return X, y


def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32, learning_rate=0.001):
    """
    训练 LSTM 模型
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 创建 DataLoader 以支持 mini-batch 训练
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    print("LSTM 模型训练完成")

    return model



# # 示例数据准备（请替换成实际时间序列数据）
# time_series_data = data['target_column'].values  # 替换为实际目标列
#
# # 1. 训练 ARIMA 模型
# arima_result = arima_model(time_series_data, order=(1, 1, 1))
#
# # 2. 创建 LSTM 数据集
# n_steps = 10  # 时间步窗口大小
# X, y = create_lstm_dataset(time_series_data, n_steps=n_steps)
#
# # 3. 初始化并训练 LSTM 模型
# lstm_model = LSTMModel(input_size=1, hidden_size=50, output_size=1, num_layers=1)
# trained_lstm_model = train_lstm_model(lstm_model, X, y, epochs=50, batch_size=32)



def main():
    print("start!")



    print("end!")



if __name__ == '__main__':
    main()

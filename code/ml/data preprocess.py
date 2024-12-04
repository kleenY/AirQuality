import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# a) 数据探索

# i. 使用Pandas等工具加载数据并查看基本信息

# Load the dataset
file_path = "./air+quality/AirQualityUCI.xlsx"
data = pd.read_excel(file_path)

# 查看数据的前5行
print("数据的前5行：")
print(data.head())

# 查看数据的基本信息
print("\n数据的信息：")
print(data.info())

# 查看数据的统计描述
print("\n数据的统计描述：")
print(data.describe())


# ii. 绘制数据分布图、时间序列图等，了解数据特征
# 将'Date'和'Time'合并为一个日期时间列
data['DateTime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str), format='%Y-%m-%d %H:%M:%S', dayfirst=True)

# 设置'DateTime'为索引
data.set_index('DateTime', inplace=True)

# 删除原始的'Date'和'Time'列
data.drop(['Date', 'Time'], axis=1, inplace=True)


# 绘制数值型数据的直方图
num_cols = data.columns

"""
for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], bins=50, kde=True)
    plt.title(f'{col} distribution map')
    plt.show()

# 绘制时间序列图
for col in num_cols:
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[col])
    plt.title(f'{col} time series map')
    plt.xlabel('DateTime')
    plt.ylabel(col)
    plt.show()
"""


# b) 数据清洗

# i. 处理缺失值：决定是删除还是填补

# 检查缺失值
print("\n缺失值统计：")
print(data.isnull().sum())

# 替换-200为NaN（根据数据说明，-200表示缺失值）
data.replace(-200, np.nan, inplace=True)

# 再次检查缺失值
print("\n替换-200后缺失值统计：")
print(data.isnull().sum())

# 计算每列缺失值的百分比
missing_percent = data.isnull().sum() / len(data) * 100
print("\n每列缺失值的百分比：")
print(missing_percent)

# 填充缺失值，这里选择使用前向填充方法
# data.fillna(method='ffill', inplace=True)

# 检查是否还有缺失值
print("\n填充后缺失值统计：")
print(data.isnull().sum())


# ii. 处理异常值：检测并处理离群点
"""
# 使用箱线图检测异常值
for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data[col])
    plt.title(f'{col} box plot')
    plt.show()
"""

# 这里选择使用IQR方法处理异常值
for col in num_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    # 定义上下界
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 截取在上下界之间的数据
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]


"""
# 使用箱线图检测处理后的数据
for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data[col])
    plt.title(f'washed {col} box plot')
    plt.show()

for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], bins=50, kde=True)
    plt.title(f'washed {col} distribution map')
    plt.show()

# 绘制时间序列图
for col in num_cols:
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[col])
    plt.title(f'washed {col} time series map')
    plt.xlabel('DateTime')
    plt.ylabel(col)
    plt.show()

"""


# iii. 数据格式转换：确保日期、时间等字段格式一致
# 已在前面将'Date'和'Time'合并为'DateTime'，并设置为索引

# c) 数据转换

# i. 对数值型数据进行标准化或归一化处理

# 初始化标准化器
scaler = StandardScaler()

# 对数值型数据进行标准化
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)


# ii. 编码分类变量（如果有）
# 本数据集中没有分类变量，如有可使用以下代码进行独热编码
# data = pd.get_dummies(data, columns=['categorical_column_name'])

# 2. 特征工程

# a) 特征提取

# i. 从时间戳中提取新的时间特征（如小时、星期几、月份、季节）

# 提取小时
data_scaled['Hour'] = data_scaled.index.hour

# 提取星期几
data_scaled['Weekday'] = data_scaled.index.weekday

# 提取月份
data_scaled['Month'] = data_scaled.index.month

# 定义季节映射函数
def get_season(month):
    if month in [12, 1, 2]:
        return 1  # 冬季
    elif month in [3, 4, 5]:
        return 2  # 春季
    elif month in [6, 7, 8]:
        return 3  # 夏季
    else:
        return 4  # 秋季

# 提取季节
data_scaled['Season'] = data_scaled.index.month.map(get_season)


# b) 特征选择

# i. 计算特征之间的相关性，选择与目标变量高度相关的特征

# 计算特征之间的相关性矩阵
corr_matrix = data_scaled.corr()

# 绘制相关性矩阵的热力图
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Heat map of correlation between features')
plt.show()



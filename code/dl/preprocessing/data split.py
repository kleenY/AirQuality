import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split



# Load the dataset and apply user-suggested processing steps
file_path = "./air+quality/AirQualityUCI.xlsx"
 
data = pd.read_excel(file_path)

# Combine 'Date' and 'Time' into a single 'DateTime' column
data['DateTime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str), format='%Y-%m-%d %H:%M:%S', dayfirst=True)

# Set 'DateTime' as the index
data.set_index('DateTime', inplace=True)

# Drop the original 'Date' and 'Time' columns
data.drop(['Date', 'Time'], axis=1, inplace=True)

# Replace -200 with NaN, indicating missing values
data.replace(-200, np.nan, inplace=True)

# Use linear interpolation to fill missing values
data.interpolate(method='linear', inplace=True)

# Verify the data
print(data.head(), data.isnull().sum())


# 重新定义滑动窗口函数
def sliding_window(data, input_hours, predict_hours, step_size):
    input_steps = input_hours  # 输入序列的小时数
    predict_steps = predict_hours  # 预测的小时数
    step = step_size  # 窗口滑动步长

    inputs = []
    labels = []
    for i in range(0, len(data) - input_steps - predict_steps + 1, step):
        input_window = data.iloc[i:i + input_steps]
        label_window = data.iloc[i + input_steps:i + input_steps + predict_steps]
        inputs.append(input_window)
        labels.append(label_window)

    return inputs, labels

# 定义分割模式
split_modes = {
    '短期': (4, 1, 2),  
    '中期': (12, 3, 2), 
    '长期': (24, 6, 2) }

# 创建保存分割数据的目录
output_dir = "./sp_data/split_data"
os.makedirs(output_dir, exist_ok=True)

# 对每种分割模式应用滑动窗口并保存结果
for mode, (input_hours, predict_hours, step_size) in split_modes.items():
    inputs, labels = sliding_window(data, input_hours, predict_hours, step_size)
    
    # 将输入和标签组合起来，形成完整的数据集
    combined = list(zip(inputs, labels))
    
    # 将数据集分割为训练集（80%）和测试集（20%）
    train_set, test_set = train_test_split(combined, test_size=0.2, random_state=42)
    
    # 保存训练集和测试集
    train_filename = os.path.join(output_dir, f"{mode}_train.pkl")
    test_filename = os.path.join(output_dir, f"{mode}_test.pkl")
    
    pd.to_pickle(train_set, train_filename)
    pd.to_pickle(test_set, test_filename)

output_dir

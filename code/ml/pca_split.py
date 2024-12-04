import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 加载用户上传的已处理数据
file_path = "./feature/NOx(GT)_processed_data.csv"
data = pd.read_csv(file_path)

# 查看数据的基本情况
data.head(), data.info()

# 首先，将时间戳列转换为时间格式并设置为索引
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S')
data.set_index('timestamp', inplace=True)

# 定义新的滑动窗口函数，用于多对一预测
def sliding_window_multi_to_one(data, input_hours, predict_hours, step_size):
    input_steps = input_hours  # 输入序列的小时数
    predict_steps = predict_hours  # 预测的小时数
    step = step_size  # 窗口滑动步长

    inputs = []
    labels = []
    for i in range(0, len(data) - input_steps - predict_steps + 1, step):
        input_window = data.iloc[i:i + input_steps, :8]   # 选择哪几列作为输入特征
        label_window = data.iloc[i + input_steps:i + input_steps + predict_steps, 0]  # 标签为第1列
        inputs.append(input_window)
        labels.append(label_window)

    return inputs, labels

# 定义分割模式
split_modes = {
    'short-term': (4, 1, 2),   # 短期模式：输入4小时，预测1小时，滑动步长为2小时
    'mid-term': (12, 3, 2),    # 中期模式：输入12小时，预测3小时，滑动步长为2小时
    'long-term': (24, 4, 2)    # 长期模式：输入24小时，预测4小时，滑动步长为2小时
}


# 创建保存分割数据的目录
output_dir = "./feature/pca_split_data"
os.makedirs(output_dir, exist_ok=True)

# 对每种分割模式应用滑动窗口并保存结果
for mode, (input_hours, predict_hours, step_size) in split_modes.items():
    inputs, labels = sliding_window_multi_to_one(data, input_hours, predict_hours, step_size)
    
    # print(f"inputs: {inputs}")
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



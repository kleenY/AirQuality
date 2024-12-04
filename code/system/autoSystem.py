import requests
import datetime
import numpy as np
import joblib
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import os
import time


API_KEY = '6bce74e98406224bffd2fb1092cb7c70'
LATITUDE = 45.4642  # City of Milan, Northern Italy 的纬度
LONGITUDE = 9.1900  # City of Milan, Northern Italy 的经度

def get_air_quality_data():
    """
    获取过去 24 小时的空气质量数据并将其转换为模型输入格式
    """
    # 获取当前时间和 24 小时前的时间戳（以秒为单位）
    current_time = int(time.time())
    start_time = current_time - 24 * 3600

    # 构造 API 请求 URL
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LATITUDE}&lon={LONGITUDE}&start={start_time}&end={current_time}&appid={API_KEY}"

    # 请求数据
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from API: {response.text}")
    
    data = response.json()

    # 初始化特征字典
    features = {
        'CO(GT)': [],
        'NO2(GT)': [],
        'NOx(GT)': [],
        'PT08.S5(O3)': [],
        'PT08.S4(NO2)': [],
        'PT08.S2(NMHC)': [],
        'PT08.S3(NOx)': [],
        'PT08.S1(CO)': []
    }

    # 解析 API 返回的数据
    for record in data.get('list', []):
        components = record['components']
        
        # 映射 API 数据到模型特征
        co = components.get('co', 0)
        no = components.get('no', 0)
        no2 = components.get('no2', 0)
        o3 = components.get('o3', 0)

        features['CO(GT)'].append(co)  # 一氧化碳
        features['NO2(GT)'].append(no2)  # 二氧化氮
        features['NOx(GT)'].append(no + no2)  # 氮氧化物（简单加和）
        features['PT08.S5(O3)'].append(o3)  # 臭氧


    return features

# 加载训练好的模型
models = {
    'Short-term': {
        'NOx(GT)': joblib.load('./model/system/random_forest_Short_NOx(GT)_model_combined.pkl'),
        'NO2(GT)': joblib.load('./model/system/random_forest_Short_NO2(GT)_model_combined.pkl'),
        'CO(GT)': joblib.load('./model/system/random_forest_Short_CO(GT)_model_combined.pkl')
    },
    'Medium-term': {
        'NOx(GT)': joblib.load('./model/system/random_forest_Mid_NOx(GT)_model_combined.pkl'),
        'NO2(GT)': joblib.load('./model/system/random_forest_Mid_NO2(GT)_model_combined.pkl'),
        'CO(GT)': joblib.load('./model/system/random_forest_Mid_CO(GT)_model_combined.pkl')
    },
    'Long-term': {
        'NOx(GT)': joblib.load('./model/systemAuto/random_forest_长期_NOx(GT)_model_combined.pkl'),
        'NO2(GT)': joblib.load('./model/systemAuto/random_forest_长期_NO2(GT)_model_combined.pkl'),
        'CO(GT)': joblib.load('./model/systemAuto/random_forest_长期_CO(GT)_model_combined.pkl')
    }
}

def predict_air_quality_via_api():
    # 获取 API 数据
    features = get_air_quality_data()

    predict_size = len(features['CO(GT)'])
    if predict_size == 24:
        mode = 'Long-term'
    else:
        return 'Invalid input length. Please input 24 values for each feature.'

    # 使用模型进行预测
    predictions = {}
    for label in ['NOx(GT)', 'NO2(GT)', 'CO(GT)']:
        input_features = np.array([features['NOx(GT)'], features['PT08.S5(O3)'], features['NO2(GT)'],
                                    features['CO(GT)']]).T

        model = models[mode][label]
        input_features = input_features.reshape(1, -1)
        y_pred = model.predict(input_features)
        predictions[label] = y_pred.flatten()

    # 绘制预测图
    figs = []
    for label in ['NOx(GT)', 'NO2(GT)', 'CO(GT)']:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(predict_size), features[label], label=f'True {label}', marker='o')
        ax.plot(range(predict_size, predict_size + predict_size // 4), predictions[label], label=f'Predicted {label}', linestyle='--', marker='x')
        ax.plot([predict_size - 1, predict_size], [features[label][-1], predictions[label][0]], linestyle='--', marker='x', color=ax.get_lines()[-1].get_color())
        ax.set_xlabel('Hours Ahead')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True)
        figs.append(fig)
    
    return figs[0], figs[1], figs[2]

# 创建 Gradio 接口
iface = gr.Interface(
    fn=predict_air_quality_via_api,
    inputs=[],
    outputs=[gr.Plot(label='NOx(GT) Prediction'), gr.Plot(label='NO2(GT) Prediction'), gr.Plot(label='CO(GT) Prediction')],
    title='Air Quality Prediction System (API Integration)',
    description='The system automatically fetches the last 24 hours of air quality data and predicts the next 6 hours.'
)

iface.launch(share=True)

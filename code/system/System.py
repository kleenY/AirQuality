import joblib
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import os

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
        'NOx(GT)': joblib.load('./model/system/random_forest_Long_NOx(GT)_model_combined.pkl'),
        'NO2(GT)': joblib.load('./model/system/random_forest_Long_NO2(GT)_model_combined.pkl'),
        'CO(GT)': joblib.load('./model/system/random_forest_Long_CO(GT)_model_combined.pkl')
    }
}

# 预测函数
def predict_air_quality(CO_GT, NO2_GT, NOx_GT, PT08_S5_O3, PT08_S4_NO2, PT08_S2_NMHC, PT08_S3_NOx, PT08_S1_CO):
    # 将输入值解析为数组
    def parse_input(input_str):
        return [float(val) for val in input_str.replace('\n', ',').split(',') if val.strip()]

    
    features = {
        'CO(GT)': parse_input(CO_GT),
        'NO2(GT)': parse_input(NO2_GT),
        'NOx(GT)': parse_input(NOx_GT),
        'PT08.S5(O3)': parse_input(PT08_S5_O3),
        'PT08.S4(NO2)': parse_input(PT08_S4_NO2),
        'PT08.S2(NMHC)': parse_input(PT08_S2_NMHC),
        'PT08.S3(NOx)': parse_input(PT08_S3_NOx),
        'PT08.S1(CO)': parse_input(PT08_S1_CO)
    }

    """
    features = {
        'CO(GT)': [float(val) for val in CO_GT.split(',')],
        'NO2(GT)': [float(val) for val in NO2_GT.split(',')],
        'NOx(GT)': [float(val) for val in NOx_GT.split(',')],
        'PT08.S5(O3)': [float(val) for val in PT08_S5_O3.split(',')],
        'PT08.S4(NO2)': [float(val) for val in PT08_S4_NO2.split(',')],
        'PT08.S2(NMHC)': [float(val) for val in PT08_S2_NMHC.split(',')],
        'PT08.S3(NOx)': [float(val) for val in PT08_S3_NOx.split(',')],
        'PT08.S1(CO)': [float(val) for val in PT08_S1_CO.split(',')]
    }"""

    # 获取输入值的数量来选择模型
    predict_size = len(features['CO(GT)'])
    if predict_size == 4:
        mode = 'Short-term'
    elif predict_size == 12:
        mode = 'Medium-term'
    elif predict_size == 24:
        mode = 'Long-term'
    else:
        return 'Invalid input length. Please input either 4, 12, or 24 values for each feature.'

    # 使用对应的模型进行预测
    predictions = {}
    for label in ['NOx(GT)', 'NO2(GT)', 'CO(GT)']:
        # 选择用于预测的特征
        input_features = np.array([features[label], features['PT08.S5(O3)'], features['PT08.S4(NO2)'],
                                    features['PT08.S2(NMHC)'], features['PT08.S3(NOx)'], features['PT08.S1(CO)']]).T

        model = models[mode][label]

        print(input_features)
        print(input_features.shape)

        input_features = input_features.reshape(1, -1)

        print(input_features)
        print(input_features.shape)

        y_pred = model.predict(input_features)
        print(f'y_pred: {y_pred}')
        print(f'y_pred shape: {y_pred.shape}')
        predictions[label] = y_pred.flatten()

    # print(f'predictions: {predictions}')
    # print(f'predictions shape: {predictions.shape}')

    
    # 绘制预测图
    figs = []
    for label in ['NOx(GT)', 'NO2(GT)', 'CO(GT)']:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(predict_size), features[label], label=f'True {label}', marker='o')
        ax.plot(range(predict_size, predict_size + predict_size // 4), predictions[label], label=f'Predicted {label}', linestyle='--', marker='x')
        # 连线最后一个特征点和第一个预测点
        ax.plot([predict_size - 1, predict_size], [features[label][-1], predictions[label][0]], linestyle='--', marker='x', color=ax.get_lines()[-1].get_color())
        ax.set_xlabel('Hours Ahead')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True)
        figs.append(fig)
    
    return figs[0], figs[1], figs[2]

# 创建 Gradio 接口
iface = gr.Interface(
    fn=predict_air_quality,
    inputs=[
        gr.Textbox(lines=1, placeholder='Enter values separated by commas (e.g. 1,2,3,4)', label='CO(GT)'),
        gr.Textbox(lines=1, placeholder='Enter values separated by commas (e.g. 1,2,3,4)', label='NO2(GT)'),
        gr.Textbox(lines=1, placeholder='Enter values separated by commas (e.g. 1,2,3,4)', label='NOx(GT)'),
        gr.Textbox(lines=1, placeholder='Enter values separated by commas (e.g. 1,2,3,4)', label='PT08.S5(O3)'),
        gr.Textbox(lines=1, placeholder='Enter values separated by commas (e.g. 1,2,3,4)', label='PT08.S4(NO2)'),
        gr.Textbox(lines=1, placeholder='Enter values separated by commas (e.g. 1,2,3,4)', label='PT08.S2(NMHC)'),
        gr.Textbox(lines=1, placeholder='Enter values separated by commas (e.g. 1,2,3,4)', label='PT08.S3(NOx)'),
        gr.Textbox(lines=1, placeholder='Enter values separated by commas (e.g. 1,2,3,4)', label='PT08.S1(CO)')
    ],
    outputs=[gr.Plot(label='NOx(GT) Prediction'), gr.Plot(label='NO2(GT) Prediction'), gr.Plot(label='CO(GT) Prediction')],
    title='Air Quality Prediction System (Demonstration)',
    description='Enter values for features and click start to predict the air quality indicators. The system can accept inputs of 4, 12, or 24 values per feature.'
)

# 启动 Gradio 接口
iface.launch(share=True)


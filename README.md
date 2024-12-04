# AirQuality
The following is the source code for our machine learning course project. The author is: Rui Hu and Kunlin Yang.

We are very sorry that we did not use Google Colab to run and save our code, as our data processing and model training procedures were carried out locally on our IDE and GPU. Therefore, the code is mostly in .py files. As a result, we have uploaded the code and data to GitHub.

# Abstract
Air pollution has become a global environmental issue, particularly severe in rapidly urbanizing regions. As one of the rapidly developing economies, the UAE faces significant air quality challenges in major cities such as Dubai and Abu Dhabi. This paper presents a machine learning-based air quality prediction model aimed at providing scientific support and technical assistance for urban air quality management. By comparing various algorithms, including Linear Regression, Random Forest, LSTM, and BiGRU, the study found that random forest outperforms others in short-term, mid-term, and long-term predictions, particularly for CO, NO2, and NOx concentrations. Additionally, we designed and implemented a real-time prediction system based on Gradio, capable of receiving current meteorological data and outputting predictions for pollutant concentrations in the coming hours. The study also included data visualization and model interpretability analysis, further enhancing the understanding of air quality trends and underlying causes.

# Directory Structure
```bash
AirQuality/
│
├── code/
│   ├── dl/                # Deep Learning Method
│   ├── ml/                # Machine Learning Method
│   └── system/            # System Source code
│
├── data/
│   ├── air+quality/       # source data
│   └── sp_data/           # Split data(preprocessed data)
│
└──  README.md             


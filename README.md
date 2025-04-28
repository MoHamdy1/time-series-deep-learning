# 💰📈 BitPredict: Bitcoin Price Forecasting

## 📖 Introduction

Bitcoin is the most widely known and traded cryptocurrency, characterized by high volatility and unpredictable price movements.  
Accurately forecasting Bitcoin prices is a challenging task due to its complex patterns, high noise, and sensitivity to global events.  
In this project, we explore multiple modeling techniques to predict Bitcoin prices based on historical data and compare deep learning models with traditional statistical methods.

---

## 🎯 Objectives

- Preprocess and visualize historical Bitcoin price data.
- Apply different forecasting techniques:
  - Dense Neural Networks
  - LSTM Networks
  - Bidirectional LSTM Networks
  - ARIMA Time Series Models
  - Facebook Prophet Model
- Evaluate and compare model performances.
- Visualize predicted vs actual Bitcoin prices.

---

## 🛠️ Models Used

| Model                        | Description |
|-------------------------------|-------------|
| Naive Forecast Model          | Baseline model to benchmark others |
| Dense Neural Network          | Fully connected simple neural network |
| LSTM                          | Recurrent network to capture long-term dependencies |
| Bi-LSTM                       | Bidirectional LSTM for better context understanding |
| ARIMA                         | Classical statistical time series model |
| Prophet                       | Forecasting tool from Facebook for trend and seasonality detection |

---

## 🧪 Approach

1. Load and preprocess Bitcoin dataset.
2. Visualize historical trends and patterns.
3. Prepare data for machine learning and time series models.
4. Train and validate each model separately.
5. Compare models based on prediction accuracy.
6. Visualize and interpret results.

---

## 📊 Results

| Model                  | MAE        | MSE         | RMSE       | MAPE    |
|-------------------------|------------|-------------|------------|---------|
| Naive Model             | 926.61     | 2.24e+06    | 1497.23    | 2.08    |
| Dense Neural Network    | 41663.69   | 2.23e+09    | 47233.50   | 48.52   |
| LSTM                    | 5272.12    | 3.67e+07    | 6062.04    | 6.30    |
| Bi-LSTM                 | 41663.69   | 2.23e+09    | 47233.50   | 48.52   |
| ARIMA (29,1,30)         | 3426.15    | 2.18e+07    | 4776.07    | 3.91    |
| ARIMA (15,1,15)         | **2935.08**| **1.96e+07**| **4422.77**| **3.36**|
| Prophet                 | 6023.91    | 4.37e+07    | 6614.39    | 7.23    |

---

## 📈 Forecast Visualization

![Forecast Graph 1](./"D:\data science projects\project 2 unfinished\time-series-deep-learning-\output.png")

---


## 🧠 Conclusion

- **ARIMA (15,1,15)** achieved the best forecasting results across all metrics.
- **LSTM** models captured some of the temporal dynamics but were less stable than ARIMA.
- **Dense Neural Networks** and **Bi-LSTM** models performed poorly without enough feature engineering.
- **Prophet** was good but less accurate compared to ARIMA and LSTM for Bitcoin's high volatility.
- Surprisingly, the **naive model** performed better than some deep models, highlighting the importance of baselines.

---

## 🔮 Future Work

- Tune LSTM/Bi-LSTM architectures with better hyperparameters.
- Add more features: technical indicators (RSI, MACD, etc.).
- Explore Transformer-based models (e.g., Time Series Transformers).
- Use ensemble models combining statistical and deep learning approaches.
- Include external data like trading volume, sentiment analysis, or macroeconomic indicators.

---

## 📚 Requirements

- Python 3.7+
- TensorFlow
- Keras
- scikit-learn
- pandas
- matplotlib
- statsmodels
- fbprophet
- seaborn

Install all dependencies:

```bash
pip install -r requirements.txt

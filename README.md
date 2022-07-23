# Stock Price Prediction Using Artificial Intelligence models

 This project aims to build a high-performance prediction model in the field of finance by applying ML/DL models. We propose "AI-Stock", a light and fast stock price forecaster which is useful in real-life applications.
 This model predicts the direction of 'long-term' and 'short-term(tomorrow)' stock prices and can be used to determine the user's investment.
 
## 1. About App

![app_example](https://user-images.githubusercontent.com/105137667/180596334-7555d270-85f3-4f0e-a72e-237c18080252.jpg)

## 2. About Data
We used stock price data from 2018-01-01 to the present from yahoo finance. Also, data highly correlated with stock prices(interest rates, commodity prices, import/export trends etc) were additionally collected.

## 3. About Models
We applied various 'Machine Learning', 'Ensemble' and 'Deep Learning' models to predict stock prices.

- Machine Learing : Linear Regression(Ridge), Support Vector Machines
- Ensemble : Random Forest, ExtraTrees, AdaBoost, XGBoost, Light GMB
- Deep Learning : Multi-Layer Perceptron(MLP), Vanila RNN, LSTM, GRU
- Transformer-based model(Doesn't apply to App) : Transformer

See "Stock Price prediction Models.ipnb" for performance evaluation of models.

## 4. About Prediction

- Short-term prediction : We use the AI models introduced above to predict the direction(up or down) of tomorrow's stock price.
- Long-term prediction : After automatically finding the optimal moving average(MA) with our model, measures trend by the "Method of Moving Average technique". Then, predicts the user's stock position(sell or buy).
 

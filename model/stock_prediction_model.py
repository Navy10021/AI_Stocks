import numpy as np
import pandas as pd
import matplotlib.pyplot as plt      
import seaborn as sns
from itertools import product

import yfinance as yf
from datetime import datetime
import datetime as dt

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

from keras.layers import Dense, Dropout, LSTM, GRU, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn.metrics import accuracy_score

class AI_Stock():
    def __init__(self, user_stock, model_name="AdaBoost", start=None, end=None):
        self.model_name = model_name
        self.models_list = {
            # 1. multi-layer-perceptron
            "MLP" : MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter=50, hidden_layer_sizes= 3*[500], random_state= 1),
            # 2. AdaBoost
            "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=1),
            # 3. ExtraTrees
            "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=1),
            # 4.RandomForest
            "RandomForest": RandomForestClassifier(max_depth=2, random_state=1),
            # 5.Support Vector Machines
            "SVM": SVC(gamma='auto'),
            # 6.XGBoost
            "XGB": lgb.LGBMClassifier(),
            # 7.Light GBM
            "LGMB": XGBClassifier(),
            # 8. Linear(Ridge)
            "Linear": RidgeClassifier(),
            # 9. LSTM
            "LSTM": None,
            # 10. GRU
            "GRU" : None,

        }
        self.model = self.models_list[model_name]

        self.stock_list = user_stock.split(",")

        if start is None:
            self.start = "2018-01-01"
        if end is None:
            self.end = datetime.now()
        
        self.raw = self.multiple_data()
        
    def multiple_data(self):
        sum_list = []
        for stock in self.stock_list:
            data = yf.download(stock, self.start, self.end)
            sum_list.append(data['Close'])
        concat_df = pd.concat(sum_list, axis=1)
        concat_df.columns = self.stock_list
        return concat_df

    def rnn_load_data(self, dataframe, lags, test_split = 0.02):
        dataframe = dataframe[['lag_'+ str(i+1) for i in range(lags)]+['direction']]
        sequence = dataframe[['lag_'+ str(i+1) for i in range(lags)]].values
        sum_num = [' '.join(str(x) for x in i) for i in sequence]
        new_df = pd.DataFrame({"sequence": sum_num})
        new_df['target'] = dataframe['direction'].values
        new_df['sequence'] = new_df['sequence'].apply(lambda x: [int(e) for e in x.split()])
        df = new_df.reindex(np.random.permutation(new_df.index))
        train_size = int(len(df) * (1 - test_split))

        X_train = df['sequence'].values[:train_size]
        y_train = np.array(df['target'].values[:train_size])
        X_test = np.array(df['sequence'].values[train_size:])
        y_test = np.array(df['target'].values[train_size:])

        return pad_sequences(X_train), y_train, pad_sequences(X_test), y_test

    def lstm_model(self, input_length):
        # 1. Create LSTM Model
        model = Sequential([
                        Embedding(input_dim=128, output_dim = 16, input_length = input_length),
                        LSTM(32, activation='sigmoid', return_sequences=True),
                        Dropout(0.5),
                        LSTM(32, activation='sigmoid', return_sequences=True),
                        LSTM(32, activation='sigmoid'),
                        Dropout(0.5),
                        Dense(1, activation='sigmoid')

        ])
        # 2. Compile
        model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
        return model

    def gru_model(self, input_length):
        # 1. Create GRU Model
        model = Sequential([
                        Embedding(input_dim=128, output_dim = 16, input_length = input_length),
                        GRU(32, activation='sigmoid', return_sequences=True),
                        Dropout(0.5),
                        GRU(32, activation='sigmoid', return_sequences=True),
                        GRU(32, activation='sigmoid'),
                        Dropout(0.5),
                        Dense(1, activation='sigmoid')

        ])
        # 2. Compile
        model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
        return model


    def rnn_predict(self, lags=5):
        acc = list()
        for symbol in self.stock_list:
            result = []
            # 1. Get one symbol data
            data = pd.DataFrame(self.raw[symbol])
            # 2. log returns
            data['returns'] = np.log(data / data.shift(1))
            data.dropna(inplace=True)
            # 3. Make lagged data
            cols = []
            for lag in range(1, lags+1):
                col = 'lag_{}'.format(lag)
                data[col] = data['returns'].shift(lag)
                cols.append(col)
                data.dropna(inplace=True)
                data[cols] = np.where(data[cols] > 0, 1, 0)
                data['direction'] = np.where(data['returns'] > 0, 1, 0)
            # 4. Make Train / Test dataset
            X_train, y_train, X_test, y_test = self.rnn_load_data(data, lags)
            # 5. Make RNN models
            if self.model_name == "LSTM":
                rnn_model = self.lstm_model(len(X_train[0]))
            else:
                rnn_model = self.gru_model(len(X_train[0]))

            hist = rnn_model.fit(X_train, y_train, batch_size=16, epochs=10, validation_split = 0.1, verbose = 0)
            test_score, test_acc = rnn_model.evaluate(X_test, y_test, batch_size=1)
            print(">> Test Acc :", test_acc)
            acc.append(test_acc)
            acc_mean = sum(acc)/len(acc)
        print(">>> Mean Accuracy Score : ", round(acc_mean, 4))

        return acc_mean


    def short_term_predict(self, lags=5, test_split=0.02):
        ans = dict()
        acc = list()
        
        for symbol in self.stock_list:
            result = []
            # 1. Get one symbol data
            data = pd.DataFrame(self.raw[symbol])
            # 2. log returns
            data['returns'] = np.log(data / data.shift(1))
            data.dropna(inplace=True)
            # 3. Make lagged data
            cols = []
            for lag in range(1, lags+1):
                col = 'lag_{}'.format(lag)
                data[col] = data['returns'].shift(lag)
                cols.append(col)
            data.dropna(inplace=True)
            # 4. Make Binary data(One-hot encoding)
            data[cols] = np.where(data[cols] > 0, 1, 0)
            # 5. Make Direction label(Up / Down)
            data['direction'] = np.where(data['returns'] > 0, 1, -1)
            
            ## SVM, MLP Ensanble Models ##
            # 6. Split dataset(train:test = 95:5)
            split = int(len(data) * (1-test_split))
            train = data.iloc[:split].copy()
            test = data.iloc[split:].copy()

            # 8. Train
            self.model.fit(train[cols], train['direction'])

            train_acc = round(accuracy_score(train['direction'], self.model.predict(train[cols])), 4)
            # 9. Test
            test['position'] = self.model.predict(test[cols])
            test_acc = round(accuracy_score(test['direction'], test['position']), 4)
            print(">> Test Acc :", test_acc)
            acc.append(test_acc)
            
            # 10. Tommarow Prediction
            input_data = pd.DataFrame(np.array([test['direction'].tail(lags)]), columns=data[cols].columns)

            if self.model.predict(input_data).tolist()[0] == 1:
                result.append("Go Up tommarow({})".format(str(self.end + dt.timedelta(days=1))[:10]))
            
            else:
                result.append("Will DROP tommarow({})".format(str(self.end + dt.timedelta(days=1))[:10]))

            ans[symbol.strip()] = result
            acc_mean = sum(acc)/len(acc)
        print(">>> Mean Accuracy Score : ", round(acc_mean, 4))

        return ans, acc_mean # {symbol = [train_acc, test_acc, prediction]}

    def long_term_predict(self):
        ans = dict()
        sma1 = range(20, 61, 4)  
        sma2 = range(180, 281, 10)

        for symbol in self.stock_list:
            result = []
            # 1. Find Optimized MA-days
            results = pd.DataFrame()
            for SMA1, SMA2 in product(sma1, sma2):
                data = pd.DataFrame(self.raw[symbol])
                data.dropna(inplace=True)
                data['Returns'] = np.log(data[symbol] / data[symbol].shift(1))
                data['Short MA'] = data[symbol].rolling(SMA1).mean()
                data['Long MA'] = data[symbol].rolling(SMA2).mean()
                data.dropna(inplace=True)
                data['Position'] = np.where(data['Short MA'] > data['Long MA'], 1, -1)
                data['Strategy'] = data['Position'].shift(1) * data['Returns']
                data.dropna(inplace=True)
                perf = np.exp(data[['Returns', 'Strategy']].sum())
                results = results.append(pd.DataFrame(
                    {'Short MA': SMA1, 'Long MA': SMA2,
                    'MARKET': perf['Returns'],
                    'STRATEGY': perf['Strategy'],
                    'OUT': perf['Strategy'] - perf['Returns']},
                    index=[0]), ignore_index=True)
                
            data = pd.DataFrame(self.raw[symbol]).dropna()
            SMA1 = int(results.sort_values('OUT', ascending=False).iloc[0]['Short MA'])
            SMA2 = int(results.sort_values('OUT', ascending=False).iloc[0]['Long MA'])
            result.append(SMA1) # Short-MA
            result.append(SMA2) # Long-MA

            # 2. Calculate Short-MA
            data['Short MA'] = data[symbol].rolling(SMA1).mean()

            # 3. Calculate Long-MA
            data['Long MA'] = data[symbol].rolling(SMA2).mean()

            # 4. Calculate your Position
            data['Position'] = np.where(data['Short MA'] > data['Long MA'], 1, -1)

            # 5. Save user position
            if data['Position'][-1] == -1:
                result.append("Long-term 'SELL' position. Make a profit or hold on")
            else:
                result.append("Long-term 'BUY' position. Buy it little by little")

            # 6. Plot
            ax = data.plot(secondary_y='Position', figsize=(12, 8))
            ax.get_legend().set_bbox_to_anchor((0.25, 0.85));
            ans[symbol.strip()] = result

        return ans  # {symbol : [short-MA, long-MA, current position]}

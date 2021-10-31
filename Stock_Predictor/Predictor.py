// This uses Keras and Tensorflow

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import datetime as dt
from pmdarima import auto_arima

class Predictor:
    _stock: pd.DataFrame = None
    _date_stock: pd.DataFrame = None
    _train = None
    _valid = None
    _NUM = 0
    _end_date = None

    def __init__(self, stock_abbreviation: str, start_date: str = None, end_date: str = None):
        if stock_abbreviation:
            stock = yf.Ticker(stock_abbreviation)
            self._stock = stock.history(start=start_date, end=end_date, actions=False)
            self._end_date = end_date
            self._NUM = self._stock.shape[0] * 95 // 100
            self._convert_to_time_series()

    @staticmethod
    def new_stock(stock_abbreviation: str = None, start_date: str = None, end_date: str = None):
        return Predictor(stock_abbreviation, start_date, end_date)

    def _convert_to_time_series(self):
        self._stock['Date'] = pd.to_datetime(self._stock.index, format="%Y-%m-%d")
        self._stock.index = self._stock['Date']
        self._stock = self._stock.sort_index(ascending=True, axis=0)

        plt.figure(figsize=(28, 12))
        plt.plot(self._stock['Open'], label='Open Price History')
        plt.legend()
        plt.show()

    def _set_date_stock(self):
        if self._date_stock is None:
            df = self._stock.copy().loc[:, ['Open', 'Date']]

            wd = []
            month = []
            year = []
            mon_fri = []

            for date in self._stock['Date']:
                dte = pd.Timestamp(date).to_pydatetime()
                day = dte.weekday()
                wd.append(day)
                if day in [0, 4]:
                    mon_fri.append(1)
                else:
                    mon_fri.append(0)
                year.append(dte.year)
                month.append(dte.month)

            df['Day_of_Week'] = wd
            df['Month'] = month
            df['Year'] = year
            df['mon_fri'] = mon_fri
            df['Quarter'] = round((df['Month'] + 1) / 3)

            self._date_stock = df.drop(['Date'], axis=1)

    @staticmethod
    def _convert_to_date_stock(date):
        date = dt.datetime.strptime(date, "%Y-%m-%d")
        date += dt.timedelta(days=1)
        arr = []
        day = date.weekday()
        arr.append(day)
        month = date.month
        arr.append(month)
        arr.append(date.year)
        if day in [0, 4]:
            arr.append(1)
        else:
            arr.append(0)
        arr.append(round((month + 1) / 3))

        return arr

    def _train_test_split(self, stock: pd.DataFrame, train_test=True):
        if 'Date' in stock.columns:
            stock.drop('Date', axis=1, inplace=True)
        if train_test:
            self._train = stock[:self._NUM]
            self._valid = stock[self._NUM:]
        else:
            self._train = stock[:]
            self._valid = None

    def _x_y_split(self, stock: pd.DataFrame, train_test=True):
        self._train_test_split(stock, train_test)

        x_train = self._train.drop('Open', axis=1)
        y_train = self._train['Open']
        if self._valid is not None:
            x_valid = self._valid.drop('Open', axis=1)
            y_valid = self._valid['Open']
            return x_train, y_train, x_valid, y_valid

        return x_train, y_train

    def _x_y_scaled_split(self, scaler, train_test=True):
        self._train_test_split(self._stock, train_test=True)

        dataset = self._stock.copy()['Open'].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(dataset)

        x_train, y_train = [], []
        for i in range(25, len(self._train)):
            x_train.append(scaled_data[i - 25:i, 0])
            y_train.append(scaled_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        return x_train, y_train

    @staticmethod
    def _calculate_rms(y_valid, prediction):
        return np.sqrt(np.mean(np.power(np.array(y_valid) - np.array(prediction), 2)))

    def linear_regression(self, train_test=True):
        self._set_date_stock()
        x_train, y_train, x_valid, y_valid = [None]*4
        data = self._x_y_split(self._date_stock, train_test)

        if train_test:
            x_train, y_train, x_valid, y_valid = data
        else:
            x_train, y_train = data

        model = LinearRegression()
        model.fit(x_train, y_train)

        if train_test:
            prediction = model.predict(x_valid)
            rms = self._calculate_rms(y_valid, prediction)
            return rms
        else:
            pred_dt = self._convert_to_date_stock(self._end_date)
            prediction = model.predict(np.reshape(pred_dt, (1, 5)))
            return prediction[0, 0]

    def lstm(self, train_test=True):
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train, y_train = self._x_y_scaled_split(scaler, train_test)

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=15, batch_size=1, verbose=2)

        x_test = []
        new_data = self._stock.copy()['Open']
        l = 1
        if train_test:
            l = len(self._valid)
        inputs = new_data[len(new_data) - l - 25:].values.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        for i in range(25, inputs.shape[0]):
            x_test.append(inputs[i - 25:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        prediction = model.predict(x_test)
        prediction = scaler.inverse_transform(prediction)

        if train_test:
            rms = self._calculate_rms(self._valid['Open'], prediction)
            return rms
        else:
            return prediction

    def auto_arima(self, train_test=True):
        self._train_test_split(self._stock, train_test)
        model = auto_arima(self._train["Open"], trace=True, error_action='ignore', start_p=1, start_q=1, max_p=10, max_q=10,
                           suppress_warnings=True, stepwise=False, seasonal=False)
        model.fit(self._train["Open"])
        if train_test:
            prediction = model.predict(n_periods=len(self._valid["Open"]))
            prediction = pd.DataFrame(prediction, index=self._valid.index, columns=['Prediction'])
            rms = self._calculate_rms(self._valid["Open"], prediction)
            return rms
        else:
            prediction = model.predict(n_periods=1)
            return prediction

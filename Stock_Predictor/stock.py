// Good Stock Predictor

import Predictor as pd
import sys


class Stock:
    _pdr = None
    _min_name = None
    _min_val = None

    def __init__(self, stock_name: str, start_date, end_date):
        self._pdr = pd.Predictor(stock_name, start_date, end_date)
        self._min_name = ''
        self._min_val = int(sys.maxsize)

    def _check_min(self, model_abbv: str, rms: int):
        if self._min_val >= rms:
            self._min_val = rms
            self._min_name = model_abbv

    def _find_best_model(self):
        rms = self._pdr.linear_regression(True)
        self._check_min('LR', rms)

        rms = self._pdr.auto_arima(True)
        self._check_min('ARIMA', rms)

        rms = self._pdr.lstm(True)
        self._check_min('LSTM', rms)

    def use_best_model(self):
        self._find_best_model()
        name = self._min_name
        predicted_value = 0
        if name is 'LR':
            predicted_value = self._pdr.linear_regression(False)
        elif name is 'ARIMA':
            predicted_value = self._pdr.auto_arima(False)
        elif name is 'LSTM':
            predicted_value = self._pdr.lstm(False)

        return predicted_value

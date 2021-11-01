# StockPredictor

```We are using Keras, Tensorflow for creating a stock predictor model.```

The date range must be ```30 days or above.```
The prediction is the Opening price of the day after the ```end_date.```


How ```stock.py``` works :

- It tests the data set using ```3 models (Linear Regression, AutoARIMA, LSTM)``` and returns the RMSE values
- Then these RMSE values are compared to find the ```best model```. That model is used to predict the Opening price.

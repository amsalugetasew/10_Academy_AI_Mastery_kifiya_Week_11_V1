import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

class ARIMAForecaster:
    def __init__(self, data, order=(1, 1, 1)):
        self.data = data
        self.data1 = self.data.asfreq('B')  # Set frequency to business days
        self.order = order
        self.model = None
        self.fitted_model = None
    
    def train_model(self):
        self.model = ARIMA(self.data1, order=self.order)
        self.fitted_model = self.model.fit()
        joblib.dump(self.fitted_model, 'arima_model.pkl')
    
    def forecast(self, steps=30):
        self.fitted_model = joblib.load('arima_model.pkl')
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        confidence_intervals = forecast_result.conf_int()
        return forecast_mean, confidence_intervals
    
    def evaluate(self, test_data):
        predictions, _ = self.forecast(steps=len(test_data))
        mae = mean_absolute_error(test_data, predictions)
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        r2 = r2_score(test_data, predictions)
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, "R Squered:": r2}
    def predict_future(self, steps=60):
        return self.forecast(steps=steps)

class SARIMAForecaster:
    def __init__(self, data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        self.data = data
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
    
    def train_model(self):
        self.model = SARIMAX(self.data, order=self.order, seasonal_order=self.seasonal_order)
        self.fitted_model = self.model.fit()
        joblib.dump(self.fitted_model, 'sarima_model.pkl')
    
    def forecast(self, steps=30):
        self.fitted_model = joblib.load('sarima_model.pkl')
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        confidence_intervals = forecast_result.conf_int()
        return forecast_mean, confidence_intervals
    
    def evaluate(self, test_data):
        predictions, _ = self.forecast(steps=len(test_data))
        mae = mean_absolute_error(test_data, predictions)
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        mae = mean_absolute_error(test_data, predictions)
        r2 = r2_score(test_data, predictions)
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, "R Squered:": r2}
    def predict_future(self, steps=60):
        return self.forecast(steps=steps)

class LSTMForecaster:
    def __init__(self, data, lstm_units=50):
        self.data = data
        self.lstm_units = lstm_units
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def train_model(self):
        scaled_data = self.scaler.fit_transform(self.data.values.reshape(-1, 1))
        X_train, y_train = [], []
        for i in range(60, len(scaled_data)):
            X_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        self.model = Sequential()
        self.model.add(LSTM(units=self.lstm_units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=self.lstm_units, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=self.lstm_units))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X_train, y_train, epochs=20, batch_size=32)
        self.model.save('lstm_model.h5')
    
    def forecast(self, steps=30):
        self.model = load_model('lstm_model.h5')
        scaled_data = self.scaler.fit_transform(self.data.values.reshape(-1, 1))
        X_input = scaled_data[-60:].reshape(1, 60, 1)
        predictions = []
        
        for _ in range(steps):
            pred = self.model.predict(X_input)
            predictions.append(pred[0, 0])
            X_input = np.append(X_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
        
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    def evaluate(self, test_data):
        predictions = self.forecast(steps=len(test_data))
        mae = mean_absolute_error(test_data, predictions)
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        r2 = r2_score(test_data, predictions)
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, "R Squered:": r2}
    def predict_future(self, steps=60):
        return self.forecast(steps=steps)


    
    
    
   

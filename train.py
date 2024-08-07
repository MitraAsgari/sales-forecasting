# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
from preprocess import load_and_preprocess_data

warnings.filterwarnings('ignore')

def train_models(df, train, test):
    # ARIMA Model
    arima_model = ARIMA(train['sales'], order=(5,1,0))
    arima_model_fit = arima_model.fit()
    arima_forecast = arima_model_fit.forecast(steps=len(test))

    # Preprocess data for XGBoost
    df = df.drop(columns=['date'])  # Drop the date column
    df = pd.get_dummies(df, columns=['family'])  # One-hot encode the 'family' column

    X = df.drop(['sales'], axis=1)
    y = df['sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    xgb_model.fit(X_train, y_train)
    xgb_forecast = xgb_model.predict(X_test)

    # LSTM Model
    sales_data = train.set_index('date')
    sales_data = sales_data.values

    train_size = int(len(sales_data) * 0.8)
    train_lstm, test_lstm = sales_data[:train_size], sales_data[train_size:]

    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    look_back = 10
    trainX, trainY = create_dataset(train_lstm, look_back)
    testX, testY = create_dataset(test_lstm, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(1, look_back)))
    lstm_model.add(LSTM(50))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

    lstm_forecast = lstm_model.predict(testX)

    # Save forecasts as .npy files
    np.save('arima_forecast.npy', arima_forecast)
    np.save('lstm_forecast.npy', lstm_forecast)
    np.save('xgb_forecast.npy', xgb_forecast)

    return arima_forecast, xgb_forecast, lstm_forecast, test['sales'].values, y_test, testY

if __name__ == "__main__":
    df, train, test = load_and_preprocess_data()
    train_models(df, train, test)

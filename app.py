# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocess import load_and_preprocess_data
from train import train_models

def evaluate_model(y_test, y_pred, model_name):
    print(f"{model_name} Evaluation:")
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))

if __name__ == "__main__":
    df, train, test = load_and_preprocess_data()
    arima_forecast, xgb_forecast, lstm_forecast, y_test_arima, y_test_xgb, y_test_lstm = train_models(df, train, test)
    
    evaluate_model(y_test_arima, arima_forecast, 'ARIMA')
    evaluate_model(y_test_xgb, xgb_forecast, 'XGBoost')
    evaluate_model(y_test_lstm, lstm_forecast, 'LSTM')

    # Visualization of results
    plt.figure(figsize=(14, 7))
    plt.plot(test['date'], test['sales'], label='Actual Sales')
    plt.plot(test['date'], arima_forecast, label='ARIMA Forecast')
    plt.plot(test['date'].iloc[10:-1], lstm_forecast.flatten(), label='LSTM Forecast')  # Adjusted length
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Sales Forecasting')
    plt.legend()
    plt.show()

    # Streamlit Dashboard
    st.title('Sales Forecasting Dashboard')
    st.line_chart(test['sales'])
    st.line_chart(arima_forecast)
    st.line_chart(lstm_forecast.flatten())
    st.write("Mean Absolute Error (ARIMA):", mean_absolute_error(y_test_arima, arima_forecast))
    st.write("Root Mean Squared Error (ARIMA):", np.sqrt(mean_squared_error(y_test_arima, arima_forecast)))
    st.write("Mean Absolute Error (XGBoost):", mean_absolute_error(y_test_xgb, xgb_forecast))
    st.write("Root Mean Squared Error (XGBoost):", np.sqrt(mean_squared_error(y_test_xgb, xgb_forecast)))
    st.write("Mean Absolute Error (LSTM):", mean_absolute_error(y_test_lstm, lstm_forecast))
    st.write("Root Mean Squared Error (LSTM):", np.sqrt(mean_squared_error(y_test_lstm, lstm_forecast)))

    # To run the Streamlit app in Colab
    os.system('streamlit run app.py & npx localtunnel --port 8501')

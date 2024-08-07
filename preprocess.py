# Import necessary libraries
import pandas as pd
import os

def download_and_extract_data():
    # Download dataset from Kaggle
    os.system('pip install -q kaggle')
    from google.colab import files
    files.upload()  # Upload your kaggle.json file

    # Create a kaggle directory and move the kaggle.json file there
    os.system('mkdir -p ~/.kaggle')
    os.system('cp kaggle.json ~/.kaggle/')
    os.system('chmod 600 ~/.kaggle/kaggle.json')

    # Download the dataset
    os.system('kaggle competitions download -c store-sales-time-series-forecasting')
    os.system('unzip -q store-sales-time-series-forecasting.zip')

def load_and_preprocess_data():
    # Load the dataset
    df = pd.read_csv('train.csv', parse_dates=['date'])
    df = df[['date', 'store_nbr', 'family', 'sales']]

    # Fill missing values if any
    df.fillna(0, inplace=True)

    # Feature Engineering
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek

    # Grouping the data by date
    sales_data = df.groupby('date')['sales'].sum().reset_index()
    
    # Splitting the dataset into training and testing sets
    train_size = int(len(sales_data) * 0.8)
    train, test = sales_data[:train_size], sales_data[train_size:]
    
    return df, train, test

if __name__ == "__main__":
    download_and_extract_data()
    df, train, test = load_and_preprocess_data()

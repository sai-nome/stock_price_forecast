import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# インタラクティブなバックエンドを設定
# matplotlib.use('TkAgg')
dm_fol = './datamart/'

import glob
stock_list = glob.glob(dm_fol + "*.parquet")

def preprocess_data(df):

    df.sort_values('Date', inplace=True)
    # 日付を特徴量に変換
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.weekday
    df['Quarter'] = df['Date'].dt.quarter

    # 周期成分のエンコーディング
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
    df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
    df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)

    # 'Date'をインデックスに設定
    df.set_index('Date', inplace=True)
    
    df.drop(columns=['symbol'], inplace=True)
    # "symbol"以外のobject型カラムをワンホットエンコーディング
    df = pd.get_dummies(df, columns=[col for col in df.select_dtypes(include=['object']).columns])

    # 全ての値が欠損している列を削除
    df.dropna(axis=1, how='all', inplace=True)

    # 値が欠損している行、infになっている行を削除
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    return df

def train_model(df):
    target = df['Close'].shift(-1).dropna()
    features = df.drop(columns=['Close'])
    features = features[:-1]

    # データのスケーリング
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    y = target.values
    
    # データを7:3に分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
    
    model = SGDRegressor(max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)
    
    return model, scaler, X_train, X_test, y_train, y_test

def update_model(model, scaler, df):
    target = df['Close'].shift(-1).dropna()
    features = df.drop(columns=['Close'])
    features = features[:-1]
    
    # データのスケーリング
    X = scaler.transform(features.values)
    y = target.values
    
    model.partial_fit(X, y)

    # 新しいデータに対する予測と評価
    # X_new = scaler.transform(new_df.drop(columns=['Close']).values)
    X_new = scaler.transform(df.drop(columns=['Close']).iloc[:-1].values)
    y_new = df['Close'].shift(-1).dropna().values
    y_new_pred = model.predict(X_new)
    mse_new = mean_squared_error(y_new, y_new_pred)
    print(f"New Data MSE: {mse_new}")

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    return y_pred

def predict_next_day(model, scaler, df):
    last_row = df.drop(columns=['Close']).iloc[-1].values.reshape(1, -1)
    # last_row = df.iloc[-1, :-1].values.reshape(1, -1)
    df.drop(columns=['Close']).iloc[-1].to_csv("test.csv")
    last_row_scaled = scaler.transform(last_row)
    next_day_prediction = model.predict(last_row_scaled)
    print(f"Previous Day Prediction: {df.iloc[-1]['Close']}")
    print(f"Next Day Prediction: {next_day_prediction[0]}")

def plot_results(df, sticker_name):
    target = df['Close'].shift(-1).dropna()
    features = df.drop(columns=['Close'])
    features = features[:-1]

    # データのスケーリング
    X = scaler.transform(features.values)
    y = target.values

    # データを7:3に分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

    y_pred = evaluate_model(model, X_test, y_test)

    plt.figure(figsize=(14, 7))
    
    # 学習データ
    train_dates = df.iloc[:len(y_train)].index
    plt.plot(train_dates, y_train, label='Train', color='red')
    
    # 検証データ（実際の値）
    test_dates = df.iloc[len(y_train):-1].index
    plt.plot(test_dates, y_test, label='Real', color='black')
    
    # 検証データ（予測値）
    plt.plot(test_dates, y_pred, label='Prediction', color='blue', linestyle='--')
    
    plt.title(sticker_name+': Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# 初回データロードとモデル学習
# start_date = '1990-04-01'
# end_date = '2024-07-03'
# symbols = ['AAPL', 'MSFT', 'GOOGL']
for count, stock_data in enumerate(stock_list):
  df = pd.read_parquet(stock_data)
  sticker_name = stock_data.split('\\')[1].replace('.parquet', '')
  df = preprocess_data(df)

#   model, scaler, X_train, X_test, y_train, y_test = train_model(df)
#   # モデルの評価
#   y_pred = evaluate_model(model, X_test, y_test)

  if count == 0:
    model, scaler, X_train, X_test, y_train, y_test = train_model(df)

    # モデルの評価
    y_pred = evaluate_model(model, X_test, y_test)

  else:

    update_model(model, scaler, df)

  # 次の日の予測
  predict_next_day(model, scaler, df)

  # 結果のプロット
  plot_results(df, sticker_name)    

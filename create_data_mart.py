import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

out_fol = "./stock_data/"
import glob
stock_list = glob.glob(out_fol + "*.csv")

def moveing_average(df):
  # 単純移動平均線（SMA）の計算
  df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()
  # 指数平滑移動平均線（EMA）の計算
  df['EMA_20'] = df['Adj Close'].ewm(span=20, adjust=False).mean()

def macd(df):
  # 短期EMA（12日）の計算
  short_ema = df['Adj Close'].ewm(span=12, adjust=False).mean()

  # 長期EMA（26日）の計算
  long_ema = df['Adj Close'].ewm(span=26, adjust=False).mean()

  # MACDラインの計算
  df['MACD'] = short_ema - long_ema

  # シグナルラインの計算（9日間のMACDのEMA）
  df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

  # MACDヒストグラムの計算
  df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']


for stock_data in stock_list:
  # データフレームの作成
  df = pd.read_csv(stock_data)
  df['Date'] = pd.to_datetime(df['Date'])
  df.set_index('Date', inplace=True)



# グラフのプロット
plt.figure(figsize=(14, 7))
plt.plot(df['Adj Close'], label='Close Price', color='black')
plt.plot(df['SMA_20'], label='20-Day SMA', color='blue')
plt.plot(df['EMA_20'], label='20-Day EMA', color='red')

plt.title('Stock Price with 20-Day SMA and 20-Day EMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

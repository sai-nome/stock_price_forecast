import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

out_fol = "./stock_data/"
import glob
stock_list = glob.glob(out_fol + "*.csv")

def moveing_average(df):
  # 単純移動平均線（SMA）の計算
  df['SMA'] = df['Adj Close'].rolling(window=20).mean()
  # 指数平滑移動平均線（EMA）の計算
  df['EMA'] = df['Adj Close'].ewm(span=20, adjust=False).mean()
  # グラフのプロット
  plt.figure(figsize=(14, 7))
  plt.plot(df['Adj Close'], label='Close Price', color='black')
  plt.plot(df['SMA'], label='20-Day SMA', color='blue')
  plt.plot(df['EMA'], label='20-Day EMA', color='red')

  plt.title('Stock Price with 20-Day SMA and 20-Day EMA')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()
  plt.grid(True)
  plt.show()

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
  # グラフのプロット
  plt.figure(figsize=(14, 7))
  # 終値のプロット
  plt.subplot(2, 1, 1)
  plt.plot(df['Close'], label='Close Price', color='black')
  plt.title('Stock Price and MACD')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()

  # MACDとシグナルラインのプロット
  plt.subplot(2, 1, 2)
  plt.plot(df['MACD'], label='MACD', color='blue')
  plt.plot(df['Signal_Line'], label='Signal Line', color='red')
  plt.bar(df.index, df['MACD_Histogram'], label='MACD Histogram', color='gray')
  plt.xlabel('Date')
  plt.ylabel('Value')
  plt.legend()

  plt.tight_layout()
  plt.show()

def bollinger_bands(df):
  # 中央バンド（20日間の単純移動平均線）の計算
  df['Middle_Band'] = df['Adj Close'].rolling(window=20).mean()

  # 標準偏差の計算
  df['STD'] = df['Adj Close'].rolling(window=20).std()

  # 上部バンドの計算
  df['Upper_Band'] = df['Middle_Band'] + (2 * df['STD'])

  # 下部バンドの計算
  df['Lower_Band'] = df['Middle_Band'] - (2 * df['STD'])

  # 過熱感の判断
  df['bands_buy_flg'] = df['Adj Close'] > df['Lower_Band']
  df['bands_sell_flg'] = df['Adj Close'] < df['Upper_Band']
  # グラフのプロット
  plt.figure(figsize=(14, 7))
  plt.plot(df['Adj Close'], label='Close Price', color='black')
  plt.plot(df['Middle_Band'], label='Middle Band (SMA 20)', color='blue')
  plt.plot(df['Upper_Band'], label='Upper Band (Middle + 2*STD)', color='red')
  plt.plot(df['Lower_Band'], label='Lower Band (Middle - 2*STD)', color='green')

  plt.fill_between(df.index, df['Upper_Band'], df['Lower_Band'], color='grey', alpha=0.1)
  plt.scatter(df.index[df['bands_buy_flg']], df['Adj Close'][df['bands_buy_flg']], label='bands_buy_flg', color='red', marker='o')
  plt.scatter(df.index[df['bands_sell_flg']], df['Adj Close'][df['bands_sell_flg']], label='bands_sell_flg', color='green', marker='o')

  plt.title('Bollinger Bands with bands_buy_flg and bands_sell_flg Signals')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()
  plt.grid(True)
  plt.show()

def ichimoku_kinko_hyo(df):
  # 転換線（過去9日間の最高値と最安値の平均値）
  df['Tenkan_sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2

  # 基準線（過去26日間の最高値と最安値の平均値）
  df['Kijun_sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2

  # 先行スパン1（転換線と基準線の平均値を26日先にプロット）
  df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)

  # 先行スパン2（過去52日間の最高値と最安値の平均値を26日先にプロット）
  df['Senkou_Span_B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)

  # 遅行スパン（現在の終値を26日遅らせてプロット）
  df['Chikou_Span'] = df['Adj Close'].shift(-26)

  # 雲の上・下の判定とシグナル
  df['Signal'] = np.where(df['Tenkan_sen'] > df['Kijun_sen'], 'Buy', 'Sell')
  df['Trend'] = np.where(df['Adj Close'] > df[['Senkou_Span_A', 'Senkou_Span_B']].max(axis=1), 'Uptrend',
                        np.where(df['Adj Close'] < df[['Senkou_Span_A', 'Senkou_Span_B']].min(axis=1), 'Downtrend', 'Neutral'))
  # グラフのプロット
  plt.figure(figsize=(14, 7))
  plt.plot(df['Adj Close'], label='Close Price', color='black')
  plt.plot(df['Tenkan_sen'], label='Tenkan-sen (Conversion Line)', color='blue')
  plt.plot(df['Kijun_sen'], label='Kijun-sen (Base Line)', color='red')
  plt.plot(df['Senkou_Span_A'], label='Senkou Span A (Leading Span 1)', color='green')
  plt.plot(df['Senkou_Span_B'], label='Senkou Span B (Leading Span 2)', color='orange')
  plt.plot(df['Chikou_Span'], label='Chikou Span (Lagging Span)', color='purple')

  # 先行スパンの領域を塗りつぶす
  plt.fill_between(df.index, df['Senkou_Span_A'], df['Senkou_Span_B'], where=df['Senkou_Span_A'] >= df['Senkou_Span_B'], color='lightgreen', alpha=0.5)
  plt.fill_between(df.index, df['Senkou_Span_A'], df['Senkou_Span_B'], where=df['Senkou_Span_A'] < df['Senkou_Span_B'], color='lightcoral', alpha=0.5)

  plt.title('Ichimoku Kinko Hyo (Ichimoku Cloud)')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()
  plt.grid(True)
  plt.show()

  # シグナルとトレンドの表示
  print(df[['Adj Close', 'Tenkan_sen', 'Kijun_sen', 'Signal', 'Trend']].tail(30))

def directional_movement_index(df):
  # True Range（TR）の計算
  df['TR'] = df[['High', 'Low', 'Adj Close']].apply(lambda x: max(x['High'] - x['Low'], abs(x['High'] - x['Adj Close']), abs(x['Low'] - x['Adj Close'])), axis=1)

  # +DMと-DMの計算
  df['+DM'] = df['High'].diff().apply(lambda x: x if x > 0 else 0)
  df['-DM'] = df['Low'].diff().apply(lambda x: -x if x < 0 else 0)

  # 14期間のTR、+DM、-DMの合計
  df['TR14'] = df['TR'].rolling(window=14).sum()
  df['+DM14'] = df['+DM'].rolling(window=14).sum()
  df['-DM14'] = df['-DM'].rolling(window=14).sum()

  # +DIと-DIの計算
  df['+DI14'] = 100 * (df['+DM14'] / df['TR14'])
  df['-DI14'] = 100 * (df['-DM14'] / df['TR14'])

  # DXの計算
  df['DX'] = 100 * (abs(df['+DI14'] - df['-DI14']) / (df['+DI14'] + df['-DI14']))

  # ADXの計算
  df['ADX'] = df['DX'].rolling(window=14).mean()

  # シグナルとトレンド強度の判定
  df['Signal'] = np.where(df['+DI14'] > df['-DI14'], 'Buy', 'Sell')
  df['Trend_Strength'] = np.where(df['ADX'] >= 25, 'Strong', 'Weak')

  # ADXのトレンド（前日と比べて上昇または下降を判定）
  df['ADX_Trend'] = df['ADX'].diff().apply(lambda x: 'Uptrend' if x > 0 else 'Downtrend' if x < 0 else 'No Change')
  # グラフのプロット
  plt.figure(figsize=(14, 7))

  plt.plot(df['+DI14'], label='+DI (Positive Directional Indicator)', color='green')
  plt.plot(df['-DI14'], label='-DI (Negative Directional Indicator)', color='red')
  plt.plot(df['ADX'], label='ADX (Average Directional Index)', color='blue')

  plt.title('DMI (Directional Movement Index)')
  plt.xlabel('Date')
  plt.ylabel('Value')
  plt.legend()
  plt.grid(True)
  plt.show()

  # 結果の表示
  print(df[['+DI14', '-DI14', 'ADX', 'Signal', 'Trend_Strength']].tail(30))

def parabolic(df):
  # パラボリックSARの計算
  df['SAR'] = np.nan
  df['EP'] = np.nan
  df['AF'] = np.nan

  # 初期値設定
  initial_trend = 'up' if df['Adj Close'].iloc[1] > df['Adj Close'].iloc[0] else 'down'
  initial_AF = 0.02
  max_AF = 0.2

  # 初期SAR、EP、AFの設定
  df.at[df.index[1], 'SAR'] = df['Low'].iloc[0] if initial_trend == 'up' else df['High'].iloc[0]
  df.at[df.index[1], 'EP'] = df['High'].iloc[1] if initial_trend == 'up' else df['Low'].iloc[1]
  df.at[df.index[1], 'AF'] = initial_AF

  # パラボリックSARの計算ループ
  for i in range(2, len(df)):
      prev_SAR = df['SAR'].iloc[i-1]
      prev_AF = df['AF'].iloc[i-1]
      prev_EP = df['EP'].iloc[i-1]
      trend = 'up' if df['Adj Close'].iloc[i-1] > prev_SAR else 'down'
      
      if trend == 'up':
          SAR = prev_SAR + prev_AF * (prev_EP - prev_SAR)
          EP = max(prev_EP, df['High'].iloc[i])
          AF = min(prev_AF + 0.02, max_AF) if df['High'].iloc[i] > prev_EP else prev_AF
      else:
          SAR = prev_SAR - prev_AF * (prev_SAR - prev_EP)
          EP = min(prev_EP, df['Low'].iloc[i])
          AF = min(prev_AF + 0.02, max_AF) if df['Low'].iloc[i] < prev_EP else prev_AF
      
      # トレンドの反転
      if trend == 'up' and df['Low'].iloc[i] < SAR:
          SAR = prev_EP
          EP = df['Low'].iloc[i]
          AF = initial_AF
          trend = 'down'
      elif trend == 'down' and df['High'].iloc[i] > SAR:
          SAR = prev_EP
          EP = df['High'].iloc[i]
          AF = initial_AF
          trend = 'up'
      
      df.at[df.index[i], 'SAR'] = SAR
      df.at[df.index[i], 'EP'] = EP
      df.at[df.index[i], 'AF'] = AF

  # シグナルの判定
  df['Signal'] = np.where((df['Adj Close'].shift(1) < df['SAR'].shift(1)) & (df['Adj Close'] > df['SAR']), 'Sell',
                          np.where((df['Adj Close'].shift(1) > df['SAR'].shift(1)) & (df['Adj Close'] < df['SAR']), 'Buy', np.nan))

  # グラフのプロット
  plt.figure(figsize=(14, 7))
  plt.plot(df['Adj Close'], label='Close Price', color='black')
  plt.plot(df['SAR'], label='Parabolic SAR', linestyle='dashed', color='blue')
  plt.scatter(df[df['Signal'] == 'Buy'].index, df[df['Signal'] == 'Buy']['Adj Close'], marker='^', color='green', label='Buy Signal', s=100)
  plt.scatter(df[df['Signal'] == 'Sell'].index, df[df['Signal'] == 'Sell']['Adj Close'], marker='v', color='red', label='Sell Signal', s=100)

  plt.title('Parabolic SAR with Buy and Sell Signals')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()
  plt.grid(True)
  plt.show()

  # 結果の表示
  print(df[['Adj Close', 'SAR', 'EP', 'AF', 'Signal']].tail(30))

def envelope(df):
  # エンベロープの計算
  percentage = 0.02  # 2% のエンベロープ
  df['Upper Envelope'] = df['SMA'] * (1 + percentage)
  df['Lower Envelope'] = df['SMA'] * (1 - percentage)
  # シグナルの判定
  df['Signal'] = np.where(df['Adj Close'] > df['Upper Envelope'], 'Sell',
                          np.where(df['Adj Close'] < df['Lower Envelope'], 'Buy', np.nan))
  # グラフのプロット
  plt.figure(figsize=(14, 7))
  plt.plot(df['Adj Close'], label='Close Price', color='black')
  plt.plot(df['SMA'], label='20-Day SMA', color='blue')
  plt.plot(df['Upper Envelope'], label='Upper Envelope (SMA + 2%)', color='green')
  plt.plot(df['Lower Envelope'], label='Lower Envelope (SMA - 2%)', color='red')
  plt.scatter(df[df['Signal'] == 'Buy'].index, df[df['Signal'] == 'Buy']['Adj Close'], marker='^', color='green', label='Buy Signal', s=100)
  plt.scatter(df[df['Signal'] == 'Sell'].index, df[df['Signal'] == 'Sell']['Adj Close'], marker='v', color='red', label='Sell Signal', s=100)

  plt.title('Envelopes with Buy and Sell Signals')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()
  plt.grid(True)
  plt.show()

  # 結果の表示
  print(df[['Adj Close', 'SMA', 'Upper Envelope', 'Lower Envelope', 'Signal']].tail(30))

def rsi(df):
  # RSIの計算
  window = 14

  # 終値の変化
  df['Change'] = df['Adj Close'].diff()

  # 上昇幅と下落幅の計算
  df['Gain'] = np.where(df['Change'] > 0, df['Change'], 0)
  df['Loss'] = np.where(df['Change'] < 0, -df['Change'], 0)

  # 平均上昇幅と平均下落幅の計算
  df['Avg_Gain'] = df['Gain'].rolling(window=window).mean()
  df['Avg_Loss'] = df['Loss'].rolling(window=window).mean()

  # 相対力（RS）の計算
  df['RS'] = df['Avg_Gain'] / df['Avg_Loss']

  # RSIの計算
  df['RSI'] = 100 - (100 / (1 + df['RS']))

  # シグナルの判定
  df['Signal'] = np.where(df['RSI'] > 70, 'Sell', np.where(df['RSI'] < 30, 'Buy', np.nan))

  # グラフのプロット
  plt.figure(figsize=(14, 7))

  # RSIのプロット
  plt.subplot(2, 1, 1)
  plt.plot(df['Adj Close'], label='Close Price', color='black')
  plt.title('Stock Price and RSI')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()

  plt.subplot(2, 1, 2)
  plt.plot(df['RSI'], label='RSI', color='blue')
  plt.axhline(70, color='red', linestyle='--')
  plt.axhline(30, color='green', linestyle='--')
  plt.scatter(df[df['Signal'] == 'Buy'].index, df[df['Signal'] == 'Buy']['RSI'], marker='^', color='green', label='Buy Signal', s=100)
  plt.scatter(df[df['Signal'] == 'Sell'].index, df[df['Signal'] == 'Sell']['RSI'], marker='v', color='red', label='Sell Signal', s=100)

  plt.title('Relative Strength Index (RSI)')
  plt.xlabel('Date')
  plt.ylabel('RSI')
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.show()

def calculate_rci(series):
    n = len(series)
    date_rank = np.arange(1, n + 1)
    price_rank = series.rank().values
    ssd = np.sum((date_rank - price_rank) ** 2)
    rci = 1 - (6 * ssd) / (n * (n ** 2 - 1))
    return rci

def rci(df):
  # RCIの計算
  window = 14

  df['RCI'] = df['Adj Close'].rolling(window=window).apply(calculate_rci, raw=False)

  # シグナルの判定（RSIと同様に70以上で売りシグナル、30以下で買いシグナル）
  df['Signal'] = np.where(df['RCI'] > 70, 'Sell', np.where(df['RCI'] < 30, 'Buy', np.nan))

  # グラフのプロット
  plt.figure(figsize=(14, 7))

  # RCIのプロット
  plt.subplot(2, 1, 1)
  plt.plot(df['Adj Close'], label='Close Price', color='black')
  plt.title('Stock Price and RCI')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()

  plt.subplot(2, 1, 2)
  plt.plot(df['RCI'], label='RCI', color='blue')
  plt.axhline(70, color='red', linestyle='--')
  plt.axhline(30, color='green', linestyle='--')
  plt.scatter(df[df['Signal'] == 'Buy'].index, df[df['Signal'] == 'Buy']['RCI'], marker='^', color='green', label='Buy Signal', s=100)
  plt.scatter(df[df['Signal'] == 'Sell'].index, df[df['Signal'] == 'Sell']['RCI'], marker='v', color='red', label='Sell Signal', s=100)

  plt.title('Rank Correlation Index (RCI)')
  plt.xlabel('Date')
  plt.ylabel('RCI')
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.show()

def madp(df):
  # 乖離率の計算
  df['madp'] = (df['Adj Close'] - df['SMA']) / df['SMA'] * 100

  # シグナルの判定
  df['Signal'] = np.nan
  threshold = 5  # 閾値の設定

  for i in range(1, len(df)):
      if df['madp'].iloc[i-1] < -threshold and df['madp'].iloc[i] > df['madp'].iloc[i-1]:
          df.at[df.index[i], 'Signal'] = 'Buy'
      elif df['madp'].iloc[i-1] > threshold and df['madp'].iloc[i] < df['madp'].iloc[i-1]:
          df.at[df.index[i], 'Signal'] = 'Sell'

  # グラフのプロット
  plt.figure(figsize=(14, 7))

  # 終値と移動平均線のプロット
  plt.subplot(2, 1, 1)
  plt.plot(df['Adj Close'], label='Close Price', color='black')
  plt.plot(df['SMA'], label='20-Day SMA', color='blue')
  plt.title('Stock Price and Moving Average')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()

  # 乖離率のプロット
  plt.subplot(2, 1, 2)
  plt.plot(df['madp'], label='madp', color='blue')
  plt.axhline(threshold, color='red', linestyle='--')
  plt.axhline(-threshold, color='green', linestyle='--')
  plt.scatter(df[df['Signal'] == 'Buy'].index, df[df['Signal'] == 'Buy']['madp'], marker='^', color='green', label='Buy Signal', s=100)
  plt.scatter(df[df['Signal'] == 'Sell'].index, df[df['Signal'] == 'Sell']['madp'], marker='v', color='red', label='Sell Signal', s=100)

  plt.title('Moving Average Convergence Divergence (MACD)')
  plt.xlabel('Date')
  plt.ylabel('madp')
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.show()

def stochastic(df):
  # %Kの計算
  window = 14
  df['14-High'] = df['High'].rolling(window=window).max()
  df['14-Low'] = df['Low'].rolling(window=window).min()
  df['%K'] = (df['Adj Close'] - df['14-Low']) / (df['14-High'] - df['14-Low']) * 100

  # スロー%Dの計算
  df['slow%D'] = df['%K'].rolling(window=3).mean()

  # シグナルの判定
  df['Signal'] = np.nan
  for i in range(1, len(df)):
      if df['slow%D'].iloc[i-1] < 20 and df['slow%D'].iloc[i] > df['slow%D'].iloc[i-1]:
          df.at[df.index[i], 'Signal'] = 'Buy'
      elif df['slow%D'].iloc[i-1] > 80 and df['slow%D'].iloc[i] < df['slow%D'].iloc[i-1]:
          df.at[df.index[i], 'Signal'] = 'Sell'

  # グラフのプロット
  plt.figure(figsize=(14, 7))

  # 終値のプロット
  plt.subplot(2, 1, 1)
  plt.plot(df['Adj Close'], label='Close Price', color='black')
  plt.title('Stock Price and Slow Stochastic')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()

  # スローストキャスティクスのプロット
  plt.subplot(2, 1, 2)
  plt.plot(df['%K'], label='%K', color='blue')
  plt.plot(df['slow%D'], label='スロー%D', color='red')
  plt.axhline(80, color='red', linestyle='--')
  plt.axhline(20, color='green', linestyle='--')
  plt.scatter(df[df['Signal'] == 'Buy'].index, df[df['Signal'] == 'Buy']['slow%D'], marker='^', color='green', label='Buy Signal', s=100)
  plt.scatter(df[df['Signal'] == 'Sell'].index, df[df['Signal'] == 'Sell']['slow%D'], marker='v', color='red', label='Sell Signal', s=100)

  plt.title('Slow Stochastic')
  plt.xlabel('Date')
  plt.ylabel('Stochastic %')
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.show()

def momentum_roc(df):
  # モメンタムとROCの計算
  window = 14
  df['Momentum'] = df['Adj Close'] - df['Adj Close'].shift(window)
  df['ROC'] = ((df['Adj Close'] - df['Adj Close'].shift(window)) / df['Adj Close'].shift(window)) * 100

  # トレンドの判定
  df['Momentum_Trend'] = np.where(df['Momentum'] > 0, 'Uptrend', np.where(df['Momentum'] < 0, 'Downtrend', 'Neutral'))
  df['ROC_Trend'] = np.where(df['ROC'] > 0, 'Uptrend', np.where(df['ROC'] < 0, 'Downtrend', 'Neutral'))

  # グラフのプロット
  plt.figure(figsize=(14, 10))

  # 終値のプロット
  plt.subplot(3, 1, 1)
  plt.plot(df['Adj Close'], label='Close Price', color='black')
  plt.title('Stock Price')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()

  # モメンタムのプロット
  plt.subplot(3, 1, 2)
  plt.plot(df['Momentum'], label='Momentum', color='blue')
  plt.axhline(0, color='black', linestyle='--')
  plt.title('Momentum')
  plt.xlabel('Date')
  plt.ylabel('Momentum')
  plt.legend()
  plt.grid(True)

  # ROCのプロット
  plt.subplot(3, 1, 3)
  plt.plot(df['ROC'], label='ROC', color='red')
  plt.axhline(0, color='black', linestyle='--')
  plt.title('Rate of Change (ROC)')
  plt.xlabel('Date')
  plt.ylabel('ROC (%)')
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.show()

def mfi(df):
  # 典型価格（TP）の計算
  df['TP'] = (df['High'] + df['Low'] + df['Adj Close']) / 3

  # マネーフロー（Raw Money Flow）の計算
  df['Raw Money Flow'] = df['TP'] * df['Volume']

  # ポジティブマネーフローとネガティブマネーフローの計算
  df['Positive Money Flow'] = np.where(df['TP'] > df['TP'].shift(1), df['Raw Money Flow'], 0)
  df['Negative Money Flow'] = np.where(df['TP'] < df['TP'].shift(1), df['Raw Money Flow'], 0)

  # 14期間の合計
  window = 14
  df['Positive Money Flow Sum'] = df['Positive Money Flow'].rolling(window=window).sum()
  df['Negative Money Flow Sum'] = df['Negative Money Flow'].rolling(window=window).sum()

  # MFIの計算
  df['Money Flow Ratio'] = df['Positive Money Flow Sum'] / df['Negative Money Flow Sum']
  df['MFI'] = 100 - (100 / (1 + df['Money Flow Ratio']))

  # シグナルの判定
  df['Signal'] = np.nan
  for i in range(1, len(df)):
      if df['MFI'].iloc[i-1] < 20 and df['MFI'].iloc[i] > df['MFI'].iloc[i-1]:
          df.at[df.index[i], 'Signal'] = 'Buy'
      elif df['MFI'].iloc[i-1] > 80 and df['MFI'].iloc[i] < df['MFI'].iloc[i-1]:
          df.at[df.index[i], 'Signal'] = 'Sell'

  # グラフのプロット
  plt.figure(figsize=(14, 7))

  # 終値のプロット
  plt.subplot(2, 1, 1)
  plt.plot(df['Adj Close'], label='Close Price', color='black')
  plt.title('Stock Price and Money Flow Index (MFI)')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()

  # MFIのプロット
  plt.subplot(2, 1, 2)
  plt.plot(df['MFI'], label='MFI', color='blue')
  plt.axhline(80, color='red', linestyle='--')
  plt.axhline(20, color='green', linestyle='--')
  plt.scatter(df[df['Signal'] == 'Buy'].index, df[df['Signal'] == 'Buy']['MFI'], marker='^', color='green', label='Buy Signal', s=100)
  plt.scatter(df[df['Signal'] == 'Sell'].index, df[df['Signal'] == 'Sell']['MFI'], marker='v', color='red', label='Sell Signal', s=100)

  plt.title('Money Flow Index (MFI)')
  plt.xlabel('Date')
  plt.ylabel('MFI')
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.show()

def cci(df):
  # 典型価格（TP）の計算
  df['TP'] = (df['High'] + df['Low'] + df['Adj Close']) / 3

  # 単純移動平均（SMA）の計算
  window = 20
  df['SMA'] = df['TP'].rolling(window=window).mean()

  # 偏差（Mean Deviation）の計算
  df['Mean Deviation'] = df['TP'].rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)

  # CCIの計算
  df['CCI'] = (df['TP'] - df['SMA']) / (0.015 * df['Mean Deviation'])

  # シグナルの判定
  df['Signal'] = np.nan
  for i in range(1, len(df)):
      if df['CCI'].iloc[i-1] < -100 and df['CCI'].iloc[i] > df['CCI'].iloc[i-1]:
          df.at[df.index[i], 'Signal'] = 'Buy'
      elif df['CCI'].iloc[i-1] > 100 and df['CCI'].iloc[i] < df['CCI'].iloc[i-1]:
          df.at[df.index[i], 'Signal'] = 'Sell'

  # グラフのプロット
  plt.figure(figsize=(14, 7))

  # 終値のプロット
  plt.subplot(2, 1, 1)
  plt.plot(df['Adj Close'], label='Close Price', color='black')
  plt.title('Stock Price and Commodity Channel Index (CCI)')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()

  # CCIのプロット
  plt.subplot(2, 1, 2)
  plt.plot(df['CCI'], label='CCI', color='blue')
  plt.axhline(100, color='red', linestyle='--')
  plt.axhline(-100, color='green', linestyle='--')
  plt.scatter(df[df['Signal'] == 'Buy'].index, df[df['Signal'] == 'Buy']['CCI'], marker='^', color='green', label='Buy Signal', s=100)
  plt.scatter(df[df['Signal'] == 'Sell'].index, df[df['Signal'] == 'Sell']['CCI'], marker='v', color='red', label='Sell Signal', s=100)

  plt.title('Commodity Channel Index (CCI)')
  plt.xlabel('Date')
  plt.ylabel('CCI')
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.show()

def gap(df):
# ギャップアップとギャップダウンの計算
  df['Previous Close'] = df['Adj Close'].shift(1)
  df['Gap'] = df['Open'] - df['Previous Close']
  df['Gap Type'] = np.where(df['Gap'] > 0, 'Gap Up', np.where(df['Gap'] < 0, 'Gap Down', 'No Gap'))

  # プロット
  plt.figure(figsize=(14, 7))

  # 終値と始値のプロット
  plt.subplot(2, 1, 1)
  plt.plot(df.index, df['Adj Close'], label='Close Price', color='black')
  plt.plot(df.index, df['Open'], label='Open Price', color='blue')
  plt.title('Stock Price with Gaps')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()

  # ギャップのプロット
  plt.subplot(2, 1, 2)
  plt.plot(df.index, df['Gap'], label='Gap', color='purple')
  plt.axhline(0, color='black', linestyle='--')
  plt.scatter(df[df['Gap Type'] == 'Gap Up'].index, df[df['Gap Type'] == 'Gap Up']['Gap'], marker='^', color='green', label='Gap Up', s=100)
  plt.scatter(df[df['Gap Type'] == 'Gap Down'].index, df[df['Gap Type'] == 'Gap Down']['Gap'], marker='v', color='red', label='Gap Down', s=100)

  plt.title('Gap Analysis')
  plt.xlabel('Date')
  plt.ylabel('Gap')
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.show()

def hv(df):
  # ヒストリカル・ボラティリティの計算
  window = 20

  # リターンの計算
  df['Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

  # 標準偏差の計算
  df['HV'] = df['Return'].rolling(window=window).std() * np.sqrt(252)

  df['RCI'] = df['Adj Close'].rolling(window=window).apply(calculate_rci, raw=False)

  # シグナルの判定
  df['Signal'] = np.nan
  for i in range(1, len(df)):
      if df['RCI'].iloc[i-1] < -80 and df['RCI'].iloc[i] > df['RCI'].iloc[i-1]:
          df.at[df.index[i], 'Signal'] = 'Buy'
      elif df['RCI'].iloc[i-1] > 80 and df['RCI'].iloc[i] < df['RCI'].iloc[i-1]:
          df.at[df.index[i], 'Signal'] = 'Sell'

  # グラフのプロット
  plt.figure(figsize=(14, 10))

  # 終値のプロット
  plt.subplot(3, 1, 1)
  plt.plot(df['Adj Close'], label='Close Price', color='black')
  plt.title('Stock Price, Historical Volatility, and RCI')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()

  # ヒストリカル・ボラティリティのプロット
  plt.subplot(3, 1, 2)
  plt.plot(df['HV'], label='Historical Volatility (HV)', color='blue')
  plt.title('Historical Volatility (HV)')
  plt.xlabel('Date')
  plt.ylabel('Volatility')
  plt.legend()
  plt.grid(True)

  # RCIのプロット
  plt.subplot(3, 1, 3)
  plt.plot(df['RCI'], label='RCI', color='red')
  plt.axhline(80, color='red', linestyle='--')
  plt.axhline(-80, color='green', linestyle='--')
  plt.scatter(df[df['Signal'] == 'Buy'].index, df[df['Signal'] == 'Buy']['RCI'], marker='^', color='green', label='Buy Signal', s=100)
  plt.scatter(df[df['Signal'] == 'Sell'].index, df[df['Signal'] == 'Sell']['RCI'], marker='v', color='red', label='Sell Signal', s=100)

  plt.title('Rank Correlation Index (RCI)')
  plt.xlabel('Date')
  plt.ylabel('RCI')
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.show()

for stock_data in stock_list:
  # データフレームの作成
  df = pd.read_csv(stock_data)
  df['Date'] = pd.to_datetime(df['Date'])
  df = df.loc[df['Date'] > '2023-06-25']
  df.set_index('Date', inplace=True)

  moveing_average(df)
  macd(df)
  bollinger_bands(df)
  ichimoku_kinko_hyo(df)
  directional_movement_index(df)
  parabolic(df)
  envelope(df)
  os.system("pause")

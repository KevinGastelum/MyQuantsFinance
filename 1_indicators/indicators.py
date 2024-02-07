import pandas as pd
import numpy as np
import pandas_ta
# LIST OF USEFUL INDICATORS

# Garman-Klass Volatility Indicator - particularly useful for assets with significant overnight price movements or markets that are open 24/7
def garman_klass_volatility(data):
    volatility = ((np.log(data['high']) - np.log(data['low'])) ** 2) / 2 - (2 * np.log(2) - 1) * ((np.log(data['adj close']) - np.log(data['open'])) ** 2)
    return volatility
# print(data)

# RSI - identify potential buying or selling opportunities, often used to determine overbought and oversold conditions
def relative_strength_index(data, length=20):
    rsi = data.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=length))
    return rsi
# print(data)
# data.xs('AAPL', level=1)['rsi'].plot()
# plt.show()

# Bollinger Bands - Primarily used to spot reversals, corrections, and potential entry/exit points based on momentum.
def bollinger_bands(data, length=20):
    bbands = data.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=length))
    bb_low = bbands.iloc[:, 0]
    bb_mid = bbands.iloc[:, 1]
    bb_high = bbands.iloc[:, 2]
    return bb_low, bb_mid, bb_high
# print(data)

# ATR- A rule of thumb is to multiply the ATR by two to determine a reasonable stop-loss point. So if you're buying a stock, you might place a stop-loss at a level twice the ATR below the entry price. If you're shorting a stock, you would place a stop-loss at a level twice the ATR *******8
def average_true_range(data, length=14):
    atr = data.groupby(level=1, group_keys=False).apply(lambda x: pandas_ta.atr(high=x['high'], low=x['low'], close=x['close'], length=length))
    return atr

# MACD - Uses two moving avgs to identify momentum and reversal points
def moving_average_convergence_divergence(data, length=20):
    macd = data.groupby(level=1, group_keys=False)['adj close'].apply(lambda x: pandas_ta.macd(close=x, fast=length).iloc[:, 0])
    return macd

# Dollar Volume - Price of stock * Volume to obtain its Market Cap
def dollar_volume(data):
    dv = (data['adj close'] * data['volume']) / 1e6
    return dv
# print(data)
# print(data.sort_values(by='dollar_volume', descending=True))


'''
====================== EXAMPLE OF USE ======================
df['garman_klass_vol'] = garman_klass_volatility(df)
df['rsi'] = relative_strength_index(df)
df[['bb_low', 'bb_mid', 'bb_high']] = bollinger_bands(df)
df['atr'] = average_true_range(df)
df['macd'] = moving_average_convergence_divergence(df)
df['dollar_volume'] = dollar_volume(df)
'''
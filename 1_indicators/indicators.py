# LIST OF USEFUL INDICATORS
import pandas as pd
import numpy as np
import pandas_ta


# Garman-Klass Volatility Indicator - particularly useful for assets with significant overnight price movements or markets that are open 24/7
def garman_klass_volatility(data):
    vol = ((np.log(data['high']) - np.log(data['low'])) ** 2) / 2 - (2 * np.log(2) - 1) * ((np.log(data['adj close']) - np.log(data['open'])) ** 2)
    return vol
df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)
# print(data)

# RSI - identify potential buying or selling opportunities, often used to determine overbought and oversold conditions
df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
# print(df)
# df.xs('AAPL', level=1)['rsi'].plot()
# plt.show()

# Bollinger Bands - Primarily used to spot reversals, corrections, and potential entry/exit points based on momentum.
df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 0])
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 1])
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 2])
# print(df)

# ATR- A rule of thumb is to multiply the ATR by two to determine a reasonable stop-loss point. So if you're buying a stock, you might place a stop-loss at a level twice the ATR below the entry price. If you're shorting a stock, you would place a stop-loss at a level twice the ATR *******8
def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'],
                      low=stock_data['low'],
                      close=stock_data['close'],
                      length=14)
    return atr.sub(atr.mean()).div(atr.std())
df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

# MACD - Uses two moving avgs to identify momentum and reversal points
def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())
df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

# Dollar Volume - Price of stock * Volume to obtain its Market Cap
df['dollar_volume'] = (df['adj close']*df['volume'])/1e6
# print(df)
# print(df.sort_values(by='dollar_volume', descending=True))
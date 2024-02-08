import ccxt
import json
import pandas as pd
import numpy as np
import pandas_ta
import os
from datetime import date, datetime, timezone, tzinfo
import time, schedule
from dotenv import load_dotenv

# =========================== LIST OF USEFUL INDICATORS =========================== #
# Garman-Klass Volatility Indicator - particularly useful for assets with significant overnight price movements or markets that are open 24/7
def garman_klass_volatility(data):
    volatility = ((np.log(data['high']) - np.log(data['low'])) ** 2) / 2 - (2 * np.log(2) - 1) * ((np.log(data['adj close']) - np.log(data['open'])) ** 2)
    return volatility
# df['garman_klass_vol'] = garman_klass_volatility(df)
# print(df)


# RSI - identify potential buying or selling opportunities, often used to determine overbought and oversold conditions
def relative_strength_index(data, length=20):
    rsi = data.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=length))
    return rsi
# df['rsi'] = relative_strength_index(df)
# print(df)
# data.xs('AAPL', level=1)['rsi'].plot()
# plt.show()


# Bollinger Bands - Primarily used to spot reversals, corrections, and potential entry/exit points based on momentum.
def bollinger_bands(data, length=20):
    bbands = data.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=length))
    bb_low = bbands.iloc[:, 0]
    bb_mid = bbands.iloc[:, 1]
    bb_high = bbands.iloc[:, 2]
    return bb_low, bb_mid, bb_high
# df[['bb_low', 'bb_mid', 'bb_high']] = bollinger_bands(df)
# print(df)


# ATR- A rule of thumb is to multiply the ATR by two to determine a reasonable stop-loss point. So if you're buying a stock, you might place a stop-loss at a level twice the ATR below the entry price. If you're shorting a stock, you would place a stop-loss at a level twice the ATR above entry
def average_true_range(data, length=14):
    atr = data.groupby(level=1, group_keys=False).apply(lambda x: pandas_ta.atr(high=x['high'], low=x['low'], close=x['close'], length=length))
    return atr
# df['atr'] = average_true_range(df)
# print(df)


# MACD - Uses two moving avgs to identify momentum and reversal points
def moving_average_convergence_divergence(data, length=20):
    macd = data.groupby(level=1, group_keys=False)['adj close'].apply(lambda x: pandas_ta.macd(close=x, fast=length).iloc[:, 0])
    return macd
# df['macd'] = moving_average_convergence_divergence(df)
# print(df)


# Dollar Volume - Price of stock * Volume to obtain its Market Cap
def dollar_volume(data):
    dv = (data['adj close'] * data['volume']) / 1e6
    return dv
# df['dollar_volume'] = dollar_volume(df)
# print(df)

'''
====================== EXAMPLE OF USE ======================
df['garman_klass_vol'] = garman_klass_volatility(df)
df['rsi'] = relative_strength_index(df)
df[['bb_low', 'bb_mid', 'bb_high']] = bollinger_bands(df)
df['atr'] = average_true_range(df)
df['macd'] = moving_average_convergence_divergence(df)
df['dollar_volume'] = dollar_volume(df)
'''



'''
Second set of indicators and functions for Algo Trading
'''
# Load env
load_dotenv()
bybt_key = os.getenv('BYBT_KEY')
bybt_secret = os.getenv('BYBT_SECRET')

# Test connections to exchange
bybit = ccxt.bybit({
  'enableRateLimit': True,
  'apiKey': bybt_key,
  'secret': bybt_secret
})
# print(bybit.fetch_balance())

# Define Constants
symbol = 'APEUSDT'
pos_size = 100
params = {'timeInForce': 'PostOnly',}
target = 35
max_loss = -55
vol_decimal = .4

# Ask or Bid function
def ask_bid(symbol=symbol):

    ob = bybit.fetch_order_book(symbol)
    # print(ob)

    bid = ob['bids'][0][0]
    ask = ob['asks'][0][0]
    print(f'This is the ask price for {symbol} {ask}')

    return ask, bid # ask_bid()[0] = ask, [1] = bid
ask_bid('BTCUSDT')

# SMA
def daily_sma(symbol=symbol):

    print('Starting Indicator...')

    timeframe = '4h'
    num_bars = 100

    bars = bybit.fetch_ohlcv(symbol, timeframe=timeframe, limit=num_bars)
    #print(bars)
    df_d = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_d['timestamp'] = pd.to_datetime(df_d['timestamp'], unit='ms')

    # DAILY SMA
    df_d['sma20_d'] = df_d.close.rolling(20).mean()

    # If bid < the 20 day SMA then = BEARISH, if bid > 20 day sma = BULLISH
    bid = ask_bid()[1]

    # If SMA > bid = SELL, if SMA < bid = BUY
    df_d.loc[df_d['sma20_d']>bid, 'sig'] = 'SELL'
    df_d.loc[df_d['sma20_d']<bid, 'sig'] = 'BUY'
    print(df_d)

    return df_d
daily_sma('BTCUSDT')
















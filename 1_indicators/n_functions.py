import ccxt
import json
import pandas as pd
import numpy as np
import pandas_ta
import os
from datetime import date, datetime, timezone, tzinfo
import time, schedule
from dotenv import load_dotenv
load_dotenv()

bybit = ccxt.bybit({
  'enableRateLimit': True,
  'apiKey': os.getenv('BYBT_KEY'),
  'secret': os.getenv('BYBT_SECRET')
})
# print(bybit.fetch_balance())

# Define Parameters
symbol = 'APEUSDT'
pos_size = 100
params = {'timeInForce': 'PostOnly',}
target = 35
max_loss = -55
vol_decimal = .4

# Dataframe param
timeframe = '4h'
limit = 100
ema = 20

# ======== Ask or Bid function ========
def ask_bid(symbol=symbol):

    ob = bybit.fetch_order_book(symbol)
    # print(ob)

    bid = ob['bids'][0][0]
    ask = ob['asks'][0][0]
    print(f'This is the ask price for {symbol} {ask}')

    return ask, bid # ask_bid()[0] = ask, [1] = bid
ask_bid('BTCUSDT')


# =========== EMA - Exponential Moving Average  ===========
def daily_ema(symbol=symbol, timeframe=timeframe, limit=limit, ema=ema):

    print('Starting Indicator...')

    bars = bybit.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    #print(bars)
    df_ema = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_ema['timestamp'] = pd.to_datetime(df_ema['timestamp'], unit='ms')

    # DAILY ema
    df_ema[f'ema{ema}_{timeframe}'] = df_ema.close.rolling(ema).mean()

    # If bid < the 20 day EMA then = BEARISH, if bid > 20 day EMA = BULLISH
    bid = ask_bid(symbol)[1]

    # If ema > bid = SELL, if ema < bid = BUY
    df_ema.loc[df_ema[f'ema{ema}_{timeframe}']>bid, 'signal'] = 'SELL'
    df_ema.loc[df_ema[f'ema{ema}_{timeframe}']<bid, 'signal'] = 'BUY'
    print(df_ema)

    return df_ema
daily_ema('BTCUSDT', '1h', 500, 200)
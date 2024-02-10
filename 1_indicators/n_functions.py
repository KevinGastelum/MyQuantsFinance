import ccxt
import json
import pandas as pd
import numpy as np
import pandas_ta # DONT NEED
import os
from datetime import date, datetime, timezone, tzinfo
import time, schedule
from dotenv import load_dotenv
load_dotenv()

bybit = ccxt.phemex({ # Add exchange function
  'enableRateLimit': True,
  'apiKey': os.getenv('PHMX_KEY'), # Add Exchange keys
  'secret': os.getenv('PHMX_SECRET')
}) # print(bybit.fetch_balance())

symbol = 'APEUSDT' # Define Parameters below
index_position = 1 # Change based on the asset
pos_size = 100
params = {'timeInForce': 'PostOnly',}
target = 35
max_loss = -55
vol_decimal = .4

# Dataframe parameters
timeframe = '4h'
limit = 100
ema = 20

#
# ======== Ask or Bid function ========
def ask_bid(symbol=symbol):

    ob = bybit.fetch_order_book(symbol)
    # print(ob)

    bid = ob['bids'][0][0]
    ask = ob['asks'][0][0]
    print(f'This is the ask price for {symbol} {ask}')

    return ask, bid # ask_bid()[0] = ask, [1] = bid
# ask_bid('BTCUSDT')


# =========== EMA - Exponential Moving Average  ===========
def df_ema(symbol=symbol, timeframe=timeframe, limit=limit, ema=ema):

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
# df_ema('BTCUSDT', '1h', 500, 200)

# =========== Open Positions (open_positions, openpos_bool, openpos_size, long)  ===========
def open_positions(index_position=index_position):
    params = {'type': 'swap', 'code': 'USD'}
    bybt_bal = bybit.fetch_balance(params=params)
    open_positions = bybt_bal['info']['data']['positions']
    # print(open_positions)

    openpos_side = open_positions[index_position]['side'] # btc [3] [0] = doge, [1] ape
    openpos_size = open_positions[index_position]['size']
    # print(open_positions)

    if openpos_side == ('Buy'):
        openpos_bool = True
        long = True
    elif openpos_side == ('Sell'):
      openpos_bool = True
      long = False
    else:
        openpos_bool = False
        long = None

    print(f'Open_positions... | openpos_bool {openpos_bool} | openpos_size {openpos_size} | long {long}')

    return open_positions, openpos_bool, openpos_size, long

open_positions()








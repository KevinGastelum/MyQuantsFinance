''' ---------------- Breakout Bot By Kevin Gastelum ----------------
- Calculates using last 3 oeef days of data
- Identifies Support and Resistance on 15m
- Exec trades on the 15m
- Place order on retest of S/R
- ticker 'uBTCUSD'
'''
import ccxt
import json
import pandas as pd
import numpy as np
import os
import sys
from datetime import date, datetime, timezone, tzinfo
import time, schedule
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '1_indicators'))
import n_functions as n
from dotenv import load_dotenv
load_dotenv()

bybit = ccxt.okx({ # Update exchange function
  # 'enableRateLimit': True,
  # 'apiKey': os.getenv('OKX_KEY'), # Add Exchange keys
  # 'secret': os.getenv('OKX_SECRET')
}) 
# print(bybit.fetchOrderBook('BTC/USDT'))
# bybit.set_sandbox_mode(True); # Enable paper trading 

symbol = 'BTCUSD' # Define Parameters below
index_pos = 1 # Index pos of asset
pause_time = 10 # The pause time between trades for Sleep function
# For Orderbook volume calcs Vol_repeat * vol_time == TIME of volume collection
vol_repeat = 11
vol_time = 5

pos_size = 10 # 125, 75, 
params = {'timeInForce': 'PostOnly',}
target = 9
max_loss = -8
vol_decimal = .4

# print(n.pnl_close('BTC/USDT'))

# Pull in the data

# Calculate Support & Resistance based on close

# Calc the retest, where we put orders

# Run bot

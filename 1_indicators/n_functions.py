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
}) 
# print(bybit.fetch_balance())

# Define Parameters below
symbol = 'APEUSDT' 
index_pos = 1 # Change based on the asset
pause_time = 60 # The pause time between trades for Sleep function
# For Orderbook volume calcs Vol_repeat * vol_time == TIME of volume collection
vol_repeat = 11
vol_time = 5

pos_size = 100
params = {'timeInForce': 'PostOnly',}
target = 35
max_loss = -55
vol_decimal = .4

# Dataframe parameters
timeframe = '4h'
limit = 100
ema = 20


# ======== Current Bid and Ask function ========
# Returns: Bid and Ask for Symbol
def ask_bid(symbol=symbol):

    ob = bybit.fetch_order_book(symbol)
    # print(ob)

    bid = ob['bids'][0][0]
    ask = ob['asks'][0][0]
    print(f'This is the ask price for {symbol} {ask}')

    return ask, bid # ask_bid()[0] = ask, [1] = bid
# ask_bid('BTCUSDT')


# =============== EMA - Exponential Moving Average  ==============
# Returns: Sell or Buy Signal into new df.columns['ema', 'signal']
def df_ema(symbol=symbol, timeframe=timeframe, limit=limit, ema=ema):

    print('Starting Indicator...')

    bars = bybit.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    #print(bars)
    df_ema = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_ema['timestamp'] = pd.to_datetime(df_ema['timestamp'], unit='ms')

    # Define ema
    df_ema[f'ema{ema}_{timeframe}'] = df_ema.close.rolling(ema).mean()

    # If bid < the 20 day EMA then = BEARISH, if bid > 20 day EMA = BULLISH
    bid = ask_bid(symbol)[1] # [1] = bid

    # If ema > bid = SELL, if ema < bid = BUY
    df_ema.loc[df_ema[f'ema{ema}_{timeframe}']>bid, 'signal'] = 'SELL'
    df_ema.loc[df_ema[f'ema{ema}_{timeframe}']<bid, 'signal'] = 'BUY'
    print(df_ema)

    return df_ema
# df_ema('BTCUSDT', '1h', 500, 200)


# ================== Fetches open positions, size, and whether long/short ==================
# Returns: Open Positions (open_positions, openpos_bool, openpos_size, long, index_position)
def open_positions(symbol=symbol):
# TODO: Figure out a way to sort through json and assign an index && Make a function that lopps through Dictionary and brings only specific coin

    # What is the position index for my symbol/ticker ## LIST YOUR ACTUAL COINS AND INDEX POSITIONS
    if symbol == 'uBTCUSD':
        index_pos = 3
    elif symbol == 'APEUSD':
        index_pos = 1
    elif symbol == 'ETHUSD':
        index_pos = 2
    elif symbol == 'SOLUSD':
        index_pos = 0
    else:
        index_pos = None

    params = {'type': 'swap', 'code': 'USD'}
    bybt_bal = bybit.fetch_balance(params=params)
    open_positions = bybt_bal['info']['data']['positions']
    # print(open_positions)

    openpos_side = open_positions[index_pos]['side'] # btc [3] [0] = doge, [1] ape
    openpos_size = open_positions[index_pos]['size']
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

    return open_positions, openpos_bool, openpos_size, long, index_pos
# open_positions()


# ====== Kill switch: pass in (symbol) if no symbol just uses default ======
# Returns: Open Positions (open_positions, openpos_bool, openpos_size, long)
def kill_switch(symbol=symbol):
    
    print(f'Starting the Kill Switch for {symbol}...')
    openposi = open_positions(symbol)[1] # openpos_bool = True or False
    long = open_positions(symbol)[3] # long = True or False
    kill_size = open_positions(symbol)[2] # openpos_size =  Size thats open

    print(f'openposi {openposi}, long {long}, size {kill_size}')

    while openposi == True:
        
        print('Starting kill switch loop til limit fill...')
        temp_df = pd.DataFrame()
        print('Creating temp df')

        # bybit.cancel_all_orders(symbol)
        openposi = open_positions(symbol)[1]
        long = open_positions(symbol)[3]
        kill_size = open_positions(symbol)[2]
        kill_size = int(kill_size)

        ask = ask_bid(symbol)[0]
        bid = ask_bid(symbol)[1]

        if long == False:
            # bybit.create_limit_buy_orders(symbol, kill_size, bid, params)
            print(f'Just made a BUY to CLOSE order of {kill_size} {symbol}, at ${bid}')
            print('Sleeping for 30secs to see if it fills...')
            time.sleep(30)
        elif long == True:
            # bybit.create_limit_buy_orders(symbol, kill_size, bid, params)
            print(f'Just made a SELL to CLOSE order of {kill_size} {symbol}, at ${ask}')
            print('Sleeping for 30secs to see if it fills...')
            time.sleep(30)
        else:
            print('+++++++ SOMETHING I DIDNT EXPECT IN KILL SWITCH FUNCTINO')

        openposi = open_positions(symbol)[1]


# ========= After closing an order we want to take a pause ========
# Returns: symbol=symbol, pause_time=pause_time -- Pause in minutes
def sleep_on_close(symbol=symbol, pause_time=pause_time):
    
    '''
    This function pulls closed orders, then if last close was in last 59min
    then it sleeps for 1min
    sincelasttrade = minutes since last trade
    '''

    closed_orders = bybit.fetch_closed_orders(symbol=symbol)
    # print(closed_orders)

    for ord in closed_orders[-1::-1]:
        
        sincelasttrade = pause_time - 1 # how long we pause

        filled = False

        status = ord['info']['ordStatus']
        txtime = ord['info']['transactTimeNs']
        txtime = int(txtime)
        txtime = round((txtime/1000000000)) # since its in nanoseconds
        print(f'For {symbol} is the status of the order {status} with epoch {txtime}')
        print('next iteration...')
        print('---------')

        if status == 'Filled':
            print('FOUND the order with the last fill...')
            print(f'For {symbol} this is the time {txtime} this is the orderstatus {status}')
            orderbook = bybit.fetch_order_book(symbol)
            ex_timestamp = orderbook['timestamp'] # in ms
            ex_timestamp = int(ex_timestamp/1000)
            print('------- Below is the transaction time and EXchange epoch time')
            print(txtime)
            print(ex_timestamp)

            time_spread = (ex_timestamp - txtime)/60

            if time_spread < sincelasttrade:
                # print('Time since last trade is less than time spread')
                # if in pos is true, put a close order here
                # if in_pos == True:

                sleepy = round(sincelasttrade-time_spread)*60
                sleepy_min = sleepy/60

                print(f'The time spread is less than {sincelasttrade} mins its been {time_spread}mins.. so we SLEEP')
                time.sleep(60)

            else:
                print(f'Its been {time_spread} mins since last fill so not sleeping, since last trade is {sincelasttrade}')
            break
        else:
            continue
# print(f'Done with the sleep on close function for {symbol}')


# ============================ Order Book Volume ==================================
# Returns: Bid/Ask Volume and appends for x iterations -- output Bearish or Bullish
def ob(symbol=symbol, vol_repeat=vol_repeat, vol_time=vol_time):
    
    print(f'Fetching order book data for {symbol}...')

    df = pd.DataFrame()
    temp_df = pd.DataFrame()

    ob = bybit.fetch_order_book(symbol)
    # print(ob)
    bids = ob['bids']
    asks = ob['asks']

    first_bid = bids[0]
    first_ask = asks[0]

    bid_vol_list = []
    ask_vol_list = []

    # if SELL vol > Buy vol AND profit target hit, exit

    # Get last x mins of volume.. and if SELL > BUY vol do x

    # repeat == the amont of times it rgoes through the vol process and multiplies by repeat time
    # repeat_time to calc the time
    for x in range(vol_repeat):
        
        for set in bids:
        # print(set)
            price = set[0]
            vol = set[1]
            bid_vol_list.append(vol)
            # print(price)
            # print(vol)

            # print(bid_vol_list)
            sum_bidvol = sum(bid_vol_list)
            # print(sum_bidvol)
            temp_df['bid_vol'] = [sum_bidvol]

        for set in asks:
            # print(set)
            price = set[0] # [40000, 344]
            vol = set[1]
            ask_vol_list.append(vol)
            # print(price)
            # print(vol)

            sum_askvol = sum(ask_vol_list)
            temp_df['ask_vol'] = [sum_askvol]
        # print(temp_df)

        time.sleep(vol_time) # Change back to 5 later
        df = pd.concat([df, temp_df], ignore_index=True) # PREVIOUSLY = df.append(temp_df)
        print(df)
        print(' ')
        print('---------')
        print(' ')
    print(f'Done collecting volume data for bid and asks...')
    print('Calculating the sums...')
    total_bidvol = df['bid_vol'].sum()
    total_askvol = df['ask_vol'].sum()
    seconds = vol_time * vol_repeat
    mins = round(seconds / 60, 2)
    print(f'Last {mins}mins for {symbol} this is total Bid Vol: {total_bidvol} | ask vol: {total_askvol}')

    if total_bidvol > total_askvol:
        control_dec = (total_askvol/total_bidvol)
        print(f'Bulls are in control, use regular target')
        # if bulls are in control use regular target
        bullish = True
    else:
        
        control_dec = (total_bidvol / total_askvol)
        print(f'Bears are in control: {control_dec}...')
        bullish = False

    # open_positions() open_positions, openpos_bool, openpos_size, long        
    open_posi = open_positions(symbol)
    openpos_tf = open_posi[1]
    long = open_posi[3]
    print(f'openpos_tf: {openpos_tf} || long: {long}')

    # if target is hit, check book vol, if book vol is < .4.. stay in pos... sleep? Need to check to see if long or short
    if openpos_tf == True:
        if long == True:
            print('We are in a long position...')
            if control_dec < vol_decimal: # vol_decimal set to .4 at top
                vol_under_dec = True
                # print('Going to sleep for a min.. since under vol decimal)
                # time.sleep(6) # Change to 60
            else:
                print('Volume is not under dec so setting vol_under_dec to False')
                vol_under_dec = False
        else:
            print('We are in a short position...')
            if control_dec < vol_decimal: # vol_decimal set to .4 at top
                vol_under_dec = True
                # print('Going to sleep for a minute.. since under vol decimal)
                # sime.sleep(6) # change to 60
            else:
                print('Volume is not under dec so setting vol_under_dec to False')
                vol_under_dec = None
    else:
        print('We are not in a position...')

    # when vol_under_dec == FALSE AND target hit, then exit
    print(vol_under_dec)

    return vol_under_dec
# For volume calcs Vol_repeat * vol_time == TIME of volume collection
# ob('uBTCUSD', 5, 1)



# pnl_close() [0] pnlclose and [1] in_pos [2]size [3]long TF
def pnl_close(symbol=symbol):

    print(f'Checking to see if its time to exit for {symbol}...')

    params = {'type':'swap', 'code':'USD'}
    pos_dict = bybit.fetch_positions(params=params)
    # print(pos_dict)

    index_pos = open_positions(symbol)[4]
    pos_dict = pos_dict[index_pos] # [3] btc [0] doge, [1] ape
    side = pos_dict['side']
    size = pos_dict['contracts']
    entry_price = float(pos_dict['entryPrice'])
    leverage = float(pos_dict['leverage'])

    current_price = ask_bid(symbol)[1]

    print(f'side: {side} | entry_price: {entry_price} | lev: {leverage}')
    # short or long

    if side == 'long':
        diff = current_price - entry_price
        long = True
    else:
        diff = entry_price - current_price
        long = False

    try:
        perc = round(((diff/entry_price) * leverage), 10)
    except:
        perc = 0

    perc = 100 * perc
    print(f'For {symbol} this is our PNL percentage: {(perc)}%')

    pnlclose = False
    in_pos = False

    if perc > 0:
        in_pos = True
        print(f'For {symbol} we are in a WINNING position')
        if perc > target:
            print('We are NOT in profit & hit target.. checking volume to see if we')
            pnlclose = True
            vol_under_dec = ob(symbol) #return TF
            if vol_under_dec == True:
                print(f'Volume is UNDER the decimal threshold we set of {vol_decimal} ')
                time.sleep(30)
            else:
                print('Starting the kill switch because we HIT our TARGET')
                # kill_switch()
        else:
            print('We have NOT hit our target yet')

    elif perc < 0: # -10, -20

        in_pos = True

        if perc <= max_loss: #under -55, -56
            print(f'We need to exit now down {perc}... so STARING the kill switch...')
            kill_switch()
        else:
            print(f'We are in a losing position of {perc}... but Max Loss NOT hit')

    else:
        print('We are not in a position')

    if in_pos == True:

        # if breaks over .8% over 15m sma, then close pos (STOP LOSS)

        # Pull in 15m ema
        df_f = df_ema(symbol, '15m', 100, 20) # df_ema(symbol, timeframe, limit, ema)
        # print(df_f)
        # df_f['ema20_15'] # last value of this
        last_ema15 = df_f.iloc[-1][f'ema{ema}_{timeframe}']
        last_ema15 = int(last_ema15)
        print(last_ema15)
        # pull current bid
        curr_bid = ask_bid(symbol)[1]
        curr_bid = int(curr_bid)
        print(curr_bid)

        sl_val = last_ema15 * 1.008
        print(sl_val)

# TURN KILL SWITCH ON

        # if curr_bid > sl_val:
        #     print('Current bid is above stop loss value.. START kill switch...')
        #     kill_switch(symbol)
        # else:
        #     print('STAYING in position')
    else:
        print('We are NOT in position...')







        print(f'For {symbol} just finished checking PNL close...')

        return pnlclose, in_pos, size, long
    
    # open_positions() 


pnl_close('BTCUSD')


'''
What I'm Building:
1. Indicators = [ Garman-Klass Volatility, RSI, Bollinger Bands, ATR, MACD, Dollar Volume ]
2. ML Unsupervised Learning Trading Strategy
3. Twitter Sentiment Trading Strategy
4. Intraday Strategy using GARCH model

Breakdown of code:
- Data used is S&P 500- 
- Indicators built here : Garman-Klass Volatility, RSI, Bollinger Bands, ATR, MACD, Dollar Volume 
- Aggreagate on the monthky level and filter for top 150 most trated stocks
- Calc monthly returns for different timeframes (1hr, 4hr, 8hr, 12hr)
- Download Fama-French Factors and caluclate rolling fctor betas for each stock
- ML - for each month fit a K-means cluster to group similar asstes based on their features
- Form a portfolio based on Efficient Frontier max sharpe ratio optimization
- Visualize (Plot) portfolio returns and compare against simply holding S&P stock
----------- By Kevin Gastelum -----------
'''

from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
# import warnings
# warnings.filterwarnings('ignore')

# STEP 1 - Download/Load S&P 500 stocks price data
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
# print(sp500)

tickers_list = sp500['Symbol'].unique().tolist()
# print(tickers_list)

end_date = '2024-02-02'
start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8)

df = yf.download(tickers=tickers_list,
                 start=start_date,
                 end=end_date).stack()
# print(df)
df.index.names = ['date', 'ticker']
df.columns = df.columns.str.lower()
# print(df)


# STEP 2  Create techincal indicators and Calculate features for each stock 
# Garman-Klass Volatility Indicator - particularly useful for assets with significant overnight price movements or markets that are open 24/7
df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)
# print(df)

# RSI - Primarily used to spot reversals, corrections, and potential entry/exit points based on momentum.
df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
# print(df)
df.xs('AAPL', level=1)['rsi'].plot()
# plt.show()

# Bollinger Bands - identify potential buying or selling opportunities, often used to determine overbought and oversold conditions
df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 0])
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 1])
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 2])
# print(df)

# ATR- A rule of thumb is to multiply the ATR by two to determine a reasonable stop-loss point. So if you're buying a stock, you might place a stop-loss at a level twice the ATR below the entry price. If you're shorting a stock, you would place a stop-loss at a level twice the ATR
def compute_atr(stock_data):
  atr = pandas_ta.atr(high=stock_data['high'],
                      low=stock_data['low'],
                      close=stock_data['close'],
                      length=14)
  return atr.sub(atr.mean()).div(atr.std())
df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)
# print(df)

# MACD
def compute_macd(close):
  macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
  return macd.sub(macd.mean()).div(macd.std())
df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)
# print(df)

# Dollar Volume - Price of stock * Volume to obtain its Market Cap
df['dollar_volume'] = (df['adj close']*df['volume'])/1e6
# print(df.sort_values(by='dollar_volume', descending=True))


# STEP 3 - Aggregate to monthly level and filter top 150 highest volume stocks for each month
# These are my feature columns [']
last_cols = [c for c in df.columns if c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]

# This uses our aggregate cols ie Indicators ['dollar_volume', 'adj close', 'garman_klass_vol', 'rsi', 'bb_low', 'bb_mid', 'bb_high', 'atr', 'macd']
data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                   df.unstack()[last_cols].resample('M').last().stack('ticker')],
                    axis=1)).dropna()
# print(data)

# Calculate the 5 year rolling avg of dollar volume for each stock before filtering
data['dollar_volume'] = (data['dollar_volume'].unstack('ticker').rolling(5*12).mean().stack())
# print(data)
# Group by month
data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))
# Filter for top 150 stocks 
data = data[data['dollar_vol_rank']<150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)
# print(data)


# STEP 4 Calculate monthly returns for different timeframes and add as features (12mos, 6mos, 1 2 3 6 9)
# g = df.xs('AAPL', level=1)
def calculate_returns(df):
    outlier_cutoff = 0.005

    lags = [1, 2, 3, 6, 9, 12]

    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close']
                            .pct_change(lag)
                            .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                   upper=x.quantile(1-outlier_cutoff)))
                            .add(1)
                            .pow(1/lag)
                            .sub(1))
    return df

data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()
# print(data)


# STEP 5 Download-Fama French Factors and Calculate Rolling Factor Betas (Risk, size, value, profitability)
# Help assess risk/return profit of portfolio - Uses RollingOLS Linear Regression
web.DataReader('F-F_Research_Data_5_Factors_2x3',
               'famafrench',
               start='2010')



















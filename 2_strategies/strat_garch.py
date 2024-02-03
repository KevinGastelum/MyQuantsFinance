'''
The 3 Projects built here
1. Unsupervised Learning Trading Strategy
2. Twitter Sentiment Trading Strategy
3. Intraday Strategy using GARCH model

Breakdown of code:
- Data used is S&P 500- 
- Indicators built here : Garman-Klass Volatility, RSI, Bollinger Bands, ATR, MACD, VWAP 
- Aggreagate on the monthky level and filter for top 150 most trated stocks
- Calc monthly returns for different timeframes
- Download Fama-French Factors and caluclate rolling fctor betas for each stock
- ML - for each month fit a K-means cluster to group similar asstes based on their features
- Form a portfolio based on Efficient Frontier max sharpe ratio optimization
- Visualize (Plot) portfolio returns and compare against simply holding S&P stock
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
import warnings
warnings.filterwarnings('ignore')

# Step 1 - Download/Load S&P 500 stocks prices data
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


# Step 2 Calculate features and techincal indicators for each stock
# Garman-Klass Volatility Indicator
df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)
# print(df)

# RSI
df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
# print(df)
df.xs('AAPL', level=1)['rsi'].plot()
# plt.show()

# Bollinger Bands
df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 0])
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 1])
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 2])
# print(df)

# ATR
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

# VWAP




'''
---------------------- By Kevin Gastelum ----------------------
What I'm Building:
1. Indicators = [ Garman-Klass Volatility, RSI, Bollinger Bands, ATR, MACD, Dollar Volume ]
2. ML Unsupervised Learning Trading Strategy
3. Twitter Sentiment Trading Strategy
4. Intraday Strategy using GARCH model

Breakdown of code:
- Data used is S&P 500
- Indicators built - Garman-Klass Volatility, RSI, Bollinger Bands, ATR, MACD, Dollar Volume 
- Aggreagate on the monthky level and filter for top 150 most trated stocks by $ volume
- Calc monthly returns for different timeframes (1mo, 3mo, 6mo, 12mo)
- Portfolio Optimization - Download Fama-French Factors and caluclate rolling fctor betas for each stock 
- ML - for each month fit a K-means cluster to group similar asstes based on their features
- Form a portfolio based on Efficient Frontier max sharpe ratio optimization
- Visualize (Plot) portfolio returns and compare against simply holding S&P stock
---------------------- By Kevin Gastelum ----------------------
'''

from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import pandas_ta
import os
import warnings
warnings.filterwarnings('ignore')

# STEP 1 - Download/Load S&P 500 stocks price data
def load_or_fetch_data(tickers, start_date, end_date, filename='sp500_data.csv'):
    if os.path.exists(filename):
        print("Loading data from local drive...")
        df = pd.read_csv(filename, header=[0, 1], index_col=0, parse_dates=True)
        # Check data is the latest
        latest_date = pd.to_datetime(df.index.max())
        if latest_date >= pd.to_datetime(end_date) - pd.DateOffset(days=1):
          return df
        else:
          print("Local data is outdated, fetching new data...")
    else:
        print("Local file not found, fetching data...")
    
    # Fetch dats if not exist or outdated
        df = yf.download(tickers=tickers, start=start_date, end=end_date)
        df.to_csv(filename)
        print("Data saved to", filename)
        return df


sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
# print(sp500)

tickers_list = sp500['Symbol'].unique().tolist()
# print(tickers_list)

end_date = '2024-02-02'
end_date = pd.to_datetime(end_date)
start_date = end_date - pd.DateOffset(365*8)

# df = yf.download(tickers=tickers_list,
#                  start=start_date,
#                  end=end_date).stack()
filename = f"sp500_data_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
df = load_or_fetch_data(tickers_list, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), filename=filename)
df = df.stack()
# print(df)
df.index.names = ['date', 'ticker']
df.columns = df.columns.str.lower()
# print(df)


# STEP 2 ============================ Building the Indicators ============================
# Garman-Klass Volatility Indicator - particularly useful for assets with significant overnight price movements or markets that are open 24/7
df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)
# print(df)

# RSI - Primarily used to spot reversals, corrections, and potential entry/exit points based on momentum.
df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
# print(df)
# df.xs('AAPL', level=1)['rsi'].plot()
# plt.show()

# Bollinger Bands - identify potential buying or selling opportunities, often used to determine overbought and oversold conditions
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


# STEP 3 ========== Aggregate on a monthly level and filter top 150 highest volume stocks per month ==============
# feature/last columns ['adj close', 'garman_klass_vol', 'rsi', 'bb_low', 'bb_mid', 'bb_high', 'atr', 'macd']
last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]
# print(last_cols)

# These are my aggregate cols i.e Indicators ['dollar_volume', 'adj close', 'garman_klass_vol', 'rsi', 'bb_low', 'bb_mid', 'bb_high', 'atr', 'macd']
data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                   df.unstack()[last_cols].resample('M').last().stack('ticker')],
                  axis=1)).dropna()
# print(data)

# Calculate the 5 year rolling avg of dollar volume for each stock before filtering
data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())
# print(data)
# Group by month
data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))
# Filter for top 150 stocks 
data = data[data['dollar_vol_rank']<150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)
data = data.sort_index(axis=1)
# print(data)


# STEP 4 ========== Calculate monthly returns for different timeframes and add as features (12mos, 6mos, 1 2 3 6 9) ==========
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


# STEP 5 ========== Download - Fama French Factors and Calculate Rolling Factor Betas (Risk, size, value, profitability) ==========
# Portfolio Optimization - Uses RollingOLS Linear Regression

factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
               'famafrench',
               start='2010')[0].drop('RF', axis=1)

factor_data.index = factor_data.index.to_timestamp()

# REfactor from percentage to decimal by dividing by 100
factor_data = factor_data.resample('M').last().div(100)
factor_data.index.name = 'date'
factor_data = factor_data.join(data['return_1m']).sort_index()
# print(factor_data.xs('AAPL', level=1).head())
# print(factor_data.info())

# Filter stocks out with less than 10 months
# print(factor_data.xs('MSFT', level=1).head())
# Groupby Months then filter stock age > 10mos
observations = factor_data.groupby(level=1).size()
valid_stocks = observations[observations >= 10]
factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]
# print(factor_data)

# Calculate Rolling Factor Betas = { 'Mkt-RF': risk, 'SMB': size, 'HML': values, 'RMW': profitablity, 'CMA': returns }
betas = (factor_data.groupby(level=1,
                            group_keys=False)
        .apply(lambda x: RollingOLS(endog=x['return_1m'],
                                    exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                    window=min(24, x.shape[0]),
                                    min_nobs=len(x.columns)+1)
        .fit(params_only=True)
        .params
        .drop('const', axis=1)))
# print(betas)

# Concat Aggregate cols with our new Fama Factors so they can begin calculating  ['garman_klass_vol', 'rsi', 'bb_low', 'bb_mid', 'bb_high', 'atr', 'macd', 'return_1m', 'return_2m', 'return_3m', 'return_6m', 'return_9m', 'return_12m', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
# Shift down Fama factors 1 row so they can calculate on the correct row
factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
data = (data.join(betas.groupby('ticker').shift()))
data.loc[:, factors] = data.groupby('ticker', group_keys=False,)[factors].apply(lambda x: x.fillna(x.mean()))
data = data.drop('adj close', axis=1)
data = data.dropna()
# print(data.info())
# print(data)


# STEP 6 ====================== Using ML model K-Means Clustering for predictions =========================
# Assign 1-4- clusters to each stock/month (4 cluster seems to be the most optimal) 

# Creates 4 clusters for each Month Then Assigns 1 optmized cluster to each stock
from sklearn.cluster import KMeans

# Create RSI targets oversold = 30 to overbought = 70
target_rsi_values = [30, 45, 55, 70]

# Define centroids above to fit in our clusters
# This ensures centroids between 30-45 are 1 cluster color and centroids between 45-55 are another etc.
initial_centroids = np.zeros((len(target_rsi_values), 18))
initial_centroids[:, 6] = target_rsi_values
# print(initial_centroids)

if 'cluster' in data.columns:
    data = data.drop('cluster', axis=1)
def get_clusters(df):
    df['cluster'] = KMeans(n_clusters=4,
                           random_state=0,
                           init=initial_centroids).fit(df).labels_ 
    return df

data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)
# print(data)

# Plot Clusters
def plot_clusters(data):
    
    cluster_0 = data[data['cluster']==0]
    cluster_1 = data[data['cluster']==1]
    cluster_2 = data[data['cluster']==2]
    cluster_3 = data[data['cluster']==3]

    plt.scatter(cluster_0.iloc[:,0], cluster_0.iloc[:,6], color='red', label='cluster 0')
    plt.scatter(cluster_1.iloc[:,0], cluster_1.iloc[:,6], color='green', label='cluster 1')
    plt.scatter(cluster_2.iloc[:,0], cluster_2.iloc[:,6], color='blue', label='cluster 2')
    plt.scatter(cluster_3.iloc[:,0], cluster_3.iloc[:,6], color='black', label='cluster 3')

    # plt.legend()
    # plt.show()
    # print(data)
    return

plt.style.use('ggplot')

for i in data.index.get_level_values('date').unique().tolist():
   g = data.xs(i, level=0)
   plt.title(f'Date {i}')
   plot_clusters(g)


# STEP 7 ====================== Portfolio Optimization with Efficient Frontier max sharpe ratio ======================
# For this strategy I will consider stocks at around the 70 RSI continue there upward momentum
filtered_df = data[data['cluster'] == 3].copy()
filtered_df = filtered_df.reset_index(level=1)
filtered_df.index = filtered_df.index+pd.DateOffset(1)
filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])
dates = filtered_df.index.get_level_values('date').unique().tolist()
fixed_dates = {}
for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()
# print(filtered_df)
# print(fixed_dates)

# Define Portfolio Optimization function
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

def optimize_weights(prices, lower_bound=0):
    returns = expected_returns.mean_historical_return(prices=prices,
                                                      frequency=252)
    cov = risk_models.sample_cov(prices=prices,
                                 frequency=252)
    
    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, .1), # Ensures portfolio is diversified 10% being the highest bound and lower bound of 2.5%
                           solver='SCS')
    
    weights = ef.max_sharpe()
    return ef.clean_weights()


stocks = data.index.get_level_values('ticker').unique().tolist()

new_df = yf.download(tickers=stocks,
                     start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                     end=data.index.get_level_values('date').unique()[-1])
# print(data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12))
# print(stocks)
# print(new_df)

returns_dataframe = np.log(new_df['Adj Close']).diff()

# Assuming new_df and returns_dataframe are already defined as per your provided code.

# Initialize an empty DataFrame for the portfolio
portfolio_df = pd.DataFrame()

# Iterate over each start date in your predetermined dates
for start_date in fixed_dates.keys():
    try:
        end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
        cols = fixed_dates[start_date]  # Columns based on the filtered tickers for the given date
        
        # Define optimization period
        optimization_start_date = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
        optimization_end_date = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        
        # Filter the adjusted close prices for the optimization period and selected tickers
        optimization_df = new_df['Adj Close'][optimization_start_date:optimization_end_date][cols]
        
        # Attempt to calculate the optimized weights
        try:
            weights = optimize_weights(optimization_df, lower_bound=round(1/(len(optimization_df.columns)*2), 3))
            weights_df = pd.DataFrame([weights], index=[start_date])  # Convert weights to DataFrame
        except Exception as e:
            print(f"Optimization failed for {start_date}: {e}. Continuing with Equal-Weights.")
            equal_weight = 1.0 / len(cols)
            weights_df = pd.DataFrame([equal_weight] * len(cols), index=cols, columns=[start_date]).T
        
        # Calculate returns for the period starting from the start_date to the end_date
        period_returns = returns_dataframe.loc[start_date:end_date][cols]
        
        # Calculate weighted returns
        weighted_returns = period_returns.mul(weights_df.iloc[0], axis=1).sum(axis=1).to_frame('Strategy Return')
        weighted_returns.index = pd.to_datetime(weighted_returns.index)

        # Append the results to the portfolio DataFrame
        portfolio_df = pd.concat([portfolio_df, weighted_returns], axis=0)
    
    except Exception as e:
        print(f"Error processing {start_date}: {e}")

#  drop duplicates or perform further cleaning on portfolio_df
portfolio_df.drop_duplicates(inplace=True)
print(portfolio_df)




spy = yf.download(tickers='SPY',
                  start='2015-01-01',
                  end=dt.date.today())

spy_ret = np.log(spy[['Adj Close']]).diff().dropna().rename({'Adj Close':'SPY Buy&Hold'}, axis=1)

portfolio_df = portfolio_df.merge(spy_ret,
                                  left_index=True,
                                  right_index=True)
# print(portfolio_df)


# STEP 8 ========================== Visualize Portfolio returns vs holding S&P500 ==========================
import matplotlib.ticker as mtick

plt.style.use('ggplot')

portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum())-1

portfolio_cumulative_return[:'2024-02-02'].plot(figsize=(16,6))

plt.title('My Quant Strategy Returns Over Time')

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

plt.ylabel('Return')

plt.show()



























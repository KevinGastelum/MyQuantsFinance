'''
The 3 Projects built here
1. Unsupervised Learning Trading Strategy
2. Twitter Sentiment Trading Strategy
3. Intraday Strategy using GARCH model

Breakdown of code:
- Data used is S&P 500- 
- Indicators built here : Bollinger Bands,  
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
import datetime as dr
import yfinance as yf
import pandas_ta
import warnings
warnings.filterwarning('ignore')


print("Hello world")



# Quants 🤖

_"A Quant, or Quantitative Analyst, embodies the fusion of finance and technology, leveraging mathematical, statistical, and computational techniques to decode the complexities of financial markets."_

<!-- The following is a brief summary of what's needed to know about the Financial Market in order to succeed as a trader. The different types of trading and strategies involved in FinTech. -->

## Demistifying the Financial Market [(FinTech)](https://en.wikipedia.org/wiki/Fintech#:~:text=Fintech%2C%20a%20clipped,fintech.%5B6%5D) 🔮

<!-- Learning about FinTech can be frustrating 😤, it's littered with complex terms designed to confuse you and scare you away. Those who do manage to navigate through the jargon, are faced with the daunting task of competing against the Market Maker i.e., Financial Institutions, Large Banks, and Firms 🏦💸. -->

Key Terms:

- **Derivatives** - AKA Options | Futures | Perpetuals 📈 - used to trade with margin (see line below).
- **Margin/Leverage** - Some exchanges allow you to borrow funds to execute trades, with the risk of being liquidated. Which occurs once your available liquidity runs out 💰.
  <br> $1,000 with 10x margin = $10,000 (Use with CAUTION ⚠️ this also means your losing money 10x faster)
- **Liquidity** (Fancy way of saying Funds) 💧
- **Assets** - Equities (Fancy name for Stocks), Commodities, Cryptocurrencies 📉📈

## Becoming a Quant 🧠💼

Knowing the basics is essential in everything, but especially when developing a winning strategy. So, I'll begin by coding the [Basic Indicators](#basic-indicators) listed below. These indicators have been used for decades BUT mastering these allows you to identify Market Trends (downtrend or uptrend), Supply and Demand zones (areas where most people bought or sold), Volume, and Volatility. From there I'll focus on creating [Trading Strategies](#trading-strategies) that leverage these insights to automize spotting entry/exit points, risk/returns, breakouts, reversals, and profits. Before executing any trade I'll run a backtest to assure a [Profitable Strategy](#profitable-quant-strategy-including-results)

<!-- risk management, portfolio optimization, profits, backtest,
Help assess risk/return profit of portfolio ,
size, value, profitability
-->

<!--
 📊
 🛒
 ⚖️
 🔊
 -->

<!-- ChatGPT help me write a -->

<!-- we will build a Backtesting script to test our results 🔄 -->

# Quant Trading 📊

### Basic Indicators:

<!-- Garman-Klass Volatility, ATR, Dollar Volume-->

1. **EMA** (Exponential Moving Average) - Weighted moving average used for trend identification 📉.
2. **RSI** (Relative Strength Index) - Momentum oscillator identifying overbought or oversold conditions 🔴🟢.
3. **VWAP** (Volume Weighted Average Price) - Average price based on volume and price 🔍.
4. **MACD** (Moving Avg Convergence Divergence) - Uses two moving avgs to identfy momentum and revrsal points.
5. **Bollinger Bands** - Used to spot reversals, corrections, and entry/exit points based on Standard deviation of EMAs.
6. **Fibonacci Retracement** - Uses Fibonacci ratios to indicate potential support or resistance levels 🔢.

### Trading Strategies

1. **Mean Reversion** - Assumes prices revert back to the mean and trades against trends 🔁.
2. **Arbitrage** - Exploits statistical mispricings of assets for profit by analyzing multiple exchanges 💹.
3. **GARCH** - Widely used for forecasting future volatility, risk management, portfolio optimization, and derivative pricing 🏃‍♂️💨.
4. **Pair Trading** - Bets on the convergence/divergence of two similar companies' stock prices 📊.
5. **Breakout Trading** - Looks for levels or areas that a stock has been unable to move beyond, and waits for it to move beyond those levels 🚪🔓.

### Advanced Trading Strategies

1. **Machine Learning Models** - Predicts market movements using historical data and algorithms; LSTM, K-Means Clustering 🤖📈.
2. **Option Strategies** - Employs methods like delta-neutral trading to hedge market movements 🛡️.
3. **Sentiment Analysis** - Analyzes market mood through news and social media for trading signals 🗞️💬.
4. **High-Frequency Trading** (HFT) - Executes numerous trades daily to capture small price movements ⚡.
5. **Market Making** - Provides market liquidity, profiting by looking at the Order book and seeing bid-ask spread to see where price is likely to gravitate to ↔️.

# Profitable Quant Strategy including results

Strategy with Results that implements the created indicators, strategies, risk management, backtest, ML for sentiment analysis and portfolio optimization

<!--

NOTES:
Identify what indicators we want to
update Mid Tier Strats



-- BASIC:
^Bollinger Bands
^ATR
^Garman-Klass Volatility - particularly useful for assets with significant overnight price movements or markets that are open 24/7
OBV

-- MID
^GARCH - Volatility Forecasting
Open Interest


===================
RISK MANAGEMENT:
Five Fama-French Factors to asses risk/return
- Market Risk
- Size
- Value
- Operating Profitability

backtesting.py

LIBRARIES
QuantLib
https://quantlib-python-docs.readthedocs.io/en/latest/


GARCH
https://arch.readthedocs.io/en/latest/univariate/introduction.html
 -->

<!-- ========================================== -->
<!-- ========================================== -->

<!--
SCRIPTS
nice funcs (indicators)
mean reversion
backtest.py
ai assistant


CONSTANTS
symbol
pos_size
params = {'timeInForce': 'PostOnly,}
target
max_loss


FUNCTIONS
ask_bid()
sma(timeframe, num_bars, bars, df, bid)
open_positions(positiions, openpos_bool, openpos_size, long, entry lev)
kill_switch(openposi, long, kil_size)
pnl_close(in_pos, size, long) ## Checks if we hit profit target or max loss
bot()


FUTURE IDEAS
Print Daily Vol in $ (sum of all big exch) = ((close price * volume) / 1e6 )
Print time in trade
 -->

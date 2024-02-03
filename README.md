# Quant 🤖💼

_"A Quant, or Quantitative Analyst, embodies the fusion of finance and technology, leveraging mathematical, statistical, and computational techniques to decode the complexities of financial markets."_

<!-- The following is a brief summary of what's needed to know about the Financial Market in order to succeed as a trader. The different types of trading and strategies involved in FinTech. -->

## Demistifying the Financial Market [(FinTech)](https://en.wikipedia.org/wiki/Fintech#:~:text=Fintech%2C%20a%20clipped,fintech.%5B6%5D) 🔮

<!-- Learning about FinTech can be frustrating 😤, it's littered with complex terms designed to confuse you and scare you away. Those who do manage to navigate through the jargon, are faced with the daunting task of competing against the Market Maker i.e., Financial Institutions, Large Banks, and Firms 🏦💸. -->

Key Terms:

- **Derivatives** - AKA Options | Futures | Perpetuals 📈 - used to trade with margin (see line below).
- **Margin/Leverage** - Some Exchanges allow you to borrow $ to execute a trade 💰.
  <br> $1,000 with 10x margin = $10,000 (Use with CAUTION ⚠️ this also means your losing money 10x faster) liquidation occurs once your liquidity runs out.
- **Liquidity** (Fancy way of saying Funds) 💧
- **Assets** - Equities (Fancy name for Stocks), Commodities, Crypto 📉📈

## Becoming a Quant 🧠

Knowing the basics is essential in everything, but especially when developing a winning strategy. The [Basic Strategies](#basic-quantitative-trading-strategies) list contains strategies that have been used for decades BUT mastering these allows you to identify key market trends and opportunities like:

- Knowing whether an asset is in an uptrend or downtrend 📊.
- Supply and Demand (These are areas where most people bought or sold) 🛒.
- Risk Management ⚖️.
- Volume and its impact on price 🔊.
<!-- - Support and Resistance -->

As I progress in my Quant journey I will begin by coding through some of these simple and commonly used indicators to automize my trading. From there I'll begin to

<!-- ChatGPT help me write a -->

<!-- we will build a Backtesting script to test our results 🔄 -->

# Trading Strategies 📊

### Basic trading strategies:

<!-- Garman-Klass Volatility, RSI, Bollinger Bands, ATR, MACD, VWAP -->

These are also commonly known as indicators, I will code these from scratch so I have

1. **EMA** (Exponential Moving Average) - Weighted moving average favoring recent prices. Used for trend identification 📉.
2. **RSI** (Relative Strength Index) - Momentum oscillator measuring speed of price movements. Identifies overbought or oversold conditions 🔴🟢.
3. **VWAP** (Volume Weighted Average Price) - Average price based on volume and price, used as a trading benchmark 🔍.
4. **MACD** (Moving Average Convergence Divergence) - Utilizes two moving averages to identify momentum and reversal points ↔️.
5. **Fibonacci Retracement** - Uses Fibonacci ratios to indicate potential support or resistance levels based 🔢.

### Mid-Level Trading Strategies

1. **Mean Reversion** - Assumes prices revert back to the mean and trades against trends 🔁.
2. **Arbitrage** - Exploits statistical mispricings of assets for profit by analyzing multiple exchanges 💹.
3. **GARCH** - Optimal for Volatility forecasting 🏃‍♂️💨.
4. **Pair Trading** - Bets on the convergence/divergence of two similar companies' stock prices 📊.
5. **Breakout Trading** - Looks for levels or areas that a stock has been unable to move beyond, and waits for it to move beyond those levels 🚪🔓.

### Advanced Trading Strategies

1. **Machine Learning Models** - Predicts market movements using historical data and algorithms; LSTM, K-Means Clustering 🤖📈.
2. **Option Strategies** - Employs methods like delta-neutral trading to hedge market movements 🛡️.
3. **Sentiment Analysis** - Analyzes market mood through news and social media for trading signals 🗞️💬.
4. **High-Frequency Trading** (HFT) - Executes numerous trades daily to capture small price movements ⚡.
5. **Market Making** - Provides market liquidity, profiting by looking at the Order book and seeing bid-ask spread to see where price is likely to gravitate to; Order Book

<!--

NOTES:
Identify what indicators we want to
update Mid Tier Strats



-- BASIC:
^Bollinger Bands
ATR
Open Interest
^Garman-Klass Volatility


-- MID
^GARCH - Volatility Forecasting


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
Print Daily Vol in $ (sum of all big exch)
Print time in trade
 -->

# Quants 🤖

_"A Quant, or Quantitative Analyst, embodies the fusion of finance and technology, leveraging mathematical, statistical, and computational techniques to decode the complexities of financial markets."_

<!-- The following is a brief summary of what's needed to know about the Financial Market in order to succeed as a trader in FinTech. -->

## Becoming a Quant ⚖️

<!-- Knowing the basics is essential in everything, but especially when developing a winning strategy.  -->

I'll begin by coding the [Basic Indicators](#basic-indicators) listed below. These indicators have been used for decades but mastering these allows you to identify Market Trend (downtrend or uptrend), Supply and Demand zones (areas where most people bought or sold), Volume, and Volatility. From there I'll focus on creating [Trading Strategies](#trading-strategies) that leverage these insights to automize spotting entry/exit points, risk/returns, breakouts, reversals, and profits. Before executing any trade live, a backtest is needed to ensure a [Profitable Strategy](#profitable-quant-strategy-results)

<!-- AI ASSISTANT - Auto GPT - Ollama - Litellm - Mistral-->
<!-- portfolio optimizations -->

## Demistifying the Financial Market [(FinTech)](https://en.wikipedia.org/wiki/Fintech#:~:text=Fintech%2C%20a%20clipped,fintech.%5B6%5D) 🔮

<!-- Market Maker i.e., Financial Institutions, Large Banks, and Firms 🏦💸. -->

Key Terms:

- **Assets** - Equities (Fancy name for Stocks), Commodities, Cryptocurrencies 📉📈
- **Liquidity** (Fancy way of saying Funds) 💧
- **Derivatives** - AKA Options | Futures | Perpetuals 📈 - used to trade with margin.
- **Margin/Leverage** - Many exchanges allow you to borrow funds to execute trades, with the risk of being liquidated, this occurs once your available liquidity runs out 💰.
  <br> $1,000 with 10x margin = $10,000 (CAUTION ⚠️ this also means you risk losing money 10x faster)

<!-- build a Backtesting script to test our results 🔄 -->

# Quant Trading 🧠💼

### Basic Indicators:

<!-- Garman-Klass Volatility, ATR, Dollar Volume-->

1. **[EMA](https://github.com/KevinGastelum/MyQuantsFinance/blob/7c416ef28ae63db1776273cb132bac01a5eebee9/1_indicators/indicators.py#L119)** (Exponential Moving Average) - Weighted moving average used for trend identification 📉.
2. **[RSI](https://github.com/KevinGastelum/MyQuantsFinance/blob/7ff0b32cc68e3cbbc401769fcd3fda117dc62380/1_indicators/indicators.py#L14)** (Relative Strength Index) - Momentum oscillator identifying overbought or oversold conditions 🔴🟢.
3. **[VWAP](https://github.com/KevinGastelum/MyQuantsFinance/blob/fa33cae78d44d042aba350a1dfe1def684cc7e87/1_indicators/indicators.py#L57)** (Volume Weighted Average Price) - Average price based on volume and price 🔍.
4. **[MACD](https://github.com/KevinGastelum/MyQuantsFinance/blob/7ff0b32cc68e3cbbc401769fcd3fda117dc62380/1_indicators/indicators.py#L43)** (Moving Avg Convergence Divergence) - Uses two moving avgs to identfy momentum and revrsal points.
5. **[Bollinger Bands](https://github.com/KevinGastelum/MyQuantsFinance/blob/7ff0b32cc68e3cbbc401769fcd3fda117dc62380/1_indicators/indicators.py#L24)** - Used to spot reversals, corrections, and entry/exit points based on Standard deviation of EMAs.
6. **Fibonacci Retracement** - Uses Fibonacci ratios to indicate potential support or resistance levels 🔢.

### Trading Strategies:

1. **Mean Reversion** - Assumes prices revert back to the mean and trades against trends 🔁.
2. **[GARCH](https://github.com/KevinGastelum/MyQuantsFinance/blob/main/2_strategies/strat_garch.py)** - Widely used for forecasting future volatility, risk management, portfolio optimization, and derivative pricing 🏃‍♂️💨.
   <img src="data\quant_strat_results.png">
3. **Pair Trading** - Bets on the convergence/divergence of two similar companies' stock prices 📊.
4. **Breakout Trading** - Looks for levels or areas that a stock has been unable to move beyond, and waits for it to move beyond those levels 🚪🔓.
5. **[Lorentzian Classifier](https://github.com/KevinGastelum/MyQuantsFinance/blob/main/2_strategies/strat_lorentzian_classifier.py)** - Uses Lorentzian distance metric and KNN algo for price forecasting.
   <!-- <img src="data\MLSentiment_spread.png"> -->
   <!-- 2. **Arbitrage** - Exploits statistical mispricings of assets for profit by analyzing multiple exchanges 💹. -->

### Advanced Trading Strategies:

1. **[Sentiment Analysis](https://github.com/KevinGastelum/MyQuantsFinance/blob/main/2_strategies/strat_sentiment_lumibot.py)** - Analyzes market mood through news and social media for trading signals 🗞️💬.
   <img src="data\MLSentiment_chart.png">
2. **Machine Learning Models** - Predicts market movements using historical data and algorithms; LSTM, K-Means Clustering 🤖📈. <!-- Linear Regression-->
3. **Option Strategies** - Employs methods like delta-neutral trading to hedge market movements 🛡️.
4. **High-Frequency Trading** (HFT) - Executes numerous trades daily to capture small price movements ⚡.
5. **Market Making** - Provides market liquidity, profiting by looking at the Order book and seeing bid-ask spread to see where price is likely to gravitate to ↔️.

# Profitable Quant Strategy results

Comparing [My Strategy](https://github.com/KevinGastelum/MyQuantsFinance/blob/main/2_strategies/strat_sentiment_lumibot.py) vs Simply holding S&P500
<img src="data\MLSentiment_spread.png">

# [Quant AI Assistant](https://github.com/KevinGastelum/MyQuantsFinance/tree/main/4_llm)

Building LLM locally to specialize in FinTech, automize analyzing research articles, data, and backtest strategies.

[Autogen](https://microsoft.github.io/autogen/docs/Getting-Started) - Create personalized agents that specialize in specific task i.e., AI Quant Research assistant

```shell
pip install autogenstudio
autogenstudio ui --port 8081 # Access AutoGenStudio in your localhost
```

[Ollama](https://github.com/ollama/ollama) - Allows you to download and run LLMs locally. <!-- curl -fsSL https://ollama.com/install.sh | sh -->

```shell
pip install ollama
ollama serve # Should run on Localhost:11434
ollama run mistral # Download & installs Mistral LLM locally ~4gb size file
```

[LiteLLM](https://litellm.ai/) - Provides embeddings, error handling, chat completion, function calling

```bash
pip install litellm
litellm --model ollama/mistral # Launches Mistral LLM locally
```

<!-- Running locally provides a cheap alternative to calling OpenAI API, -->

<!--  Share plot images -->

<!-- risk management, portfolio optimization, profits, backtest,
Help assess risk/return profit of portfolio ,
size, value, profitability
-->

<!--

TODO:
=====================================
5. Build Breakout Strat:
https://www.youtube.com/watch?v=5q6s6n1f8d8&list=PLvzuUVysUFOuoRna8KhschkVVUo2E2g6G&index=19

Lorentzian Classifier

Add AI chat images



===================

-- INDICATORS:
Kernel Regression
ATR
Garman-Klass Volatility - particularly useful for assets with significant overnight price movements or markets that are open 24/7
OBV

-- ADV
LSTM + GARCH
Linear Regression
Open Interest
Order Book
Liquidation Sniper

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
open_positions(positions, openpos_bool, openpos_size, long, entry lev)
kill_switch(openposi, long, kil_size)
pnl_close(in_pos, size, long) ## Checks if we hit profit target or max loss
bot()


FUTURE *
Print Daily Vol in $ (sum of all big exch) = ((close price * volume) / 1e6 )
Print time in trade

📊 🛒 ⚖️ 🔊

============= MOONDEV ====================
-- AI MoondDev
AI ASSISTANT - Auto GPT - Ollama - Litellm - Mistral
https://www.youtube.com/watch?v=4ZqJSfV4818&pp=ygUOYXV0b2dlbiBzdHVkaW8%3D

https://www.youtube.com/watch?v=mUEFwUU0IfE&pp=ygUOYXV0b2dlbiBzdHVkaW8%3D
 -->

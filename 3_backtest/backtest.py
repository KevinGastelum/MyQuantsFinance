import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1 insert market data
def load_data(filename):
    df = pd.read_csv(filename, parse_dates=True, index_col='Date')
    return df

# 2 Define Straetgy (The following is a basic moving average strategy)
def strategy(df, short_window, long_window):
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0

    # Create short simple moving average over the short window
    signals['short_mavg'] = df['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

    # Create long simple moving average over the long window
    signals['long_mavg'] = df['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Create signals
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                                > signals['long_mavg'][short_window:], 1.0, 0.0)   

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals


# 3 Backtest Strategy
def backtest(signals, df, initial_capital=float(100000.0)):
    portfolio = pd.DataFrame(index=signals.index)
    portfolio['holdings'] = (signals['positions'].cumsum() * df['Close'])
    portfolio['cash'] = initial_capital - (signals['positions'] * df['Close']).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()
    return portfolio


# 4 Plot results
def plot_results(df, signals, portfolio):
    # Create a plot
    fig, ax1 = plt.subplots(figsize=(12,8))

    # Plot the closing price and moving averages
    df['Close'].plot(ax=ax1, color='r', lw=2.)
    signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

    # Plot the buy signals
    ax1.plot(signals.loc[signals.positions == 1.0].index, 
             signals.short_mavg[signals.positions == 1.0],
             '^', markersize=10, color='m')
             
    # Plot the sell signals
    ax1.plot(signals.loc[signals.positions == -1.0].index, 
             signals.short_mavg[signals.positions == -1.0],
             'v', markersize=10, color='k')
             
    # Show the plot
    plt.show()

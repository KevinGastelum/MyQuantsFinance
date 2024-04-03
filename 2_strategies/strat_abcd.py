import pandas as pd
import numpy as np
from pandas import DataFrame

# from freqtrade.strategy import IStrategy


class ZigZagPAStrategyV4(IStrategy):
    useHA = False
    useAltTF = True
    tf = 60
    timeframe = "5m"
    minimal_roi = {
        "0": 0.1
    }

    # Stop Loss
    stoploss = -0.99

    def zigzag(self, dataframe: DataFrame):

        is_up_current = []

        is_down_current = []

        direction_current = [None]
        direction_previous = [None]

        zigzag = []

        for index, row in dataframe.iterrows():

            is_up = row['close'] >= row['open']
            is_down = row['close'] <= row['open']

            is_up_current.append(is_up)
            is_up_previous = is_up_current[:-1]

            is_down_current.append(is_down)
            is_down_previous = is_down_current[:-1]

            # Calculate direction
            if is_up_previous and is_down:
                direction = -1
            elif is_down_previous and is_up:
                direction = 1
            else:
                direction = direction_previous

            direction_current.append(direction)
            direction_previous = direction_current[:-1]

            # Build zigzag
            if (is_up and is_down) and direction_previous != -1:
                zigzag.append(row['high'])
            elif (is_down and is_up) and direction_previous != 1:
                zigzag.append(row['low'])
            else:
                zigzag.append(np.nan)

        return pd.DataFrame({'zigzag': zigzag})

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # ZigZag
        dataframe['zigzag'] = self.zigzag(dataframe)

        # 
        x = dataframe['zigzag'].shift(4)
        a = dataframe['zigzag'].shift(3)
        b = dataframe['zigzag'].shift(2)
        c = dataframe['zigzag'].shift(1)
        d = dataframe['zigzag']

        # 
        dataframe['xab'] = (np.abs(b - a) / np.abs(x - a))
        dataframe['xad'] = (np.abs(a - d) / np.abs(x - a))
        dataframe['abc'] = (np.abs(b - c) / np.abs(a - b))
        dataframe['bcd'] = (np.abs(c - d) / np.abs(b - c))

        # 
        dataframe['x'] = x
        dataframe['a'] = a
        dataframe['b'] = b
        dataframe['c'] = c
        dataframe['d'] = d

        return dataframe

    def f_last_fib(self, dataframe: DataFrame, rate: float) -> float:
        fib_range = abs(dataframe['d'] - dataframe['c'])
        if fib_range.all() == 0:
            return dataframe['d']
        else:
            diff = dataframe['d'] - dataframe['c']
            return dataframe['d'] - (fib_range * rate) * np.sign(diff)

    def isBat(self, dataframe: DataFrame, _mode: int = 1):
        _xab = (dataframe['xab'] >= 0.382) & (dataframe['xab'] <= 0.5)
        _abc = (dataframe['abc'] >= 0.382) & (dataframe['abc'] <= 0.886)
        _bcd = (dataframe['bcd'] >= 1.618) & (dataframe['bcd'] <= 2.618)
        _xad = (dataframe['xad'] <= 0.618) & (dataframe['xad'] <= 1.000)  # 0.886
        return (_xab & _abc & _bcd & _xad & ((_mode == 1) & (dataframe['d'] < dataframe['c'])) | (
                    (_mode != 1) & (dataframe['d'] > dataframe['c'])))

    def isAntiBat(self, dataframe: DataFrame, _mode: int = 1):
        _xab = (dataframe['xab'] >= 0.500) & (dataframe['xab'] <= 0.886)  # 0.618
        _abc = (dataframe['abc'] >= 1.000) & (dataframe['abc'] <= 2.618)  # 1.13 --> 2.618
        _bcd = (dataframe['bcd'] >= 1.618) & (dataframe['bcd'] <= 2.618)  # 2.0  --> 2.618
        _xad = (dataframe['xad'] >= 0.886) & (dataframe['xad'] <= 1.000)  # 1.13
        return (_xab & _abc & _bcd & _xad & ((_mode == 1) & (dataframe['d'] < dataframe['c'])) | (
                    (_mode != 1) & (dataframe['d'] > dataframe['c'])))

    def isAltBat(self, dataframe: DataFrame, _mode: int = 1):
        _xab = (dataframe['xab'] <= 0.382)
        _abc = (dataframe['abc'] >= 0.382) & (dataframe['abc'] <= 0.886)
        _bcd = (dataframe['bcd'] >= 2.0) & (dataframe['bcd'] <= 3.618)
        _xad = (dataframe['xad'] <= 1.13)
        return (_xab & _abc & _bcd & _xad & ((_mode == 1) & (dataframe['d'] < dataframe['c'])) | (
                    (_mode != 1) & (dataframe['d'] > dataframe['c'])))

    def isButterfly(self, dataframe: DataFrame, _mode: int = 1):
        _xab = (dataframe['xab'] <= 0.786)
        _abc = (dataframe['abc'] >= 0.382) & (dataframe['abc'] <= 0.886)
        _bcd = (dataframe['bcd'] >= 1.618) & (dataframe['bcd'] <= 2.618)
        _xad = (dataframe['xad'] >= 1.27) & (dataframe['xad'] <= 1.618)
        return (_xab & _abc & _bcd & _xad & ((_mode == 1) & (dataframe['d'] < dataframe['c'])) | (
                    (_mode != 1) & (dataframe['d'] > dataframe['c'])))

    def isAntiButterfly(self, dataframe: DataFrame, _mode: int = 1):
        _xab = (dataframe['xab'] >= 0.236) & (dataframe['xab'] <= 0.886)  # 0.382 - 0.618
        _abc = (dataframe['abc'] >= 1.130) & (dataframe['abc'] <= 2.618)  # 1.130 - 2.618
        _bcd = (dataframe['bcd'] >= 1.000) & (dataframe['bcd'] <= 1.382)  # 1.27
        _xad = (dataframe['xad'] >= 0.500) & (dataframe['xad'] <= 0.886)  # 0.618 - 0.786
        return (_xab & _abc & _bcd & _xad & ((_mode == 1) & (dataframe['d'] < dataframe['c'])) | (
                    (_mode != 1) & (dataframe['d'] > dataframe['c'])))

    def isABCD(self, dataframe: DataFrame, _mode: int = 1):
        _abc = (dataframe['abc'] >= 0.382) & (dataframe['abc'] <= 0.886)
        _bcd = (dataframe['bcd'] >= 1.13) & (dataframe['bcd'] <= 2.618)
        return (_abc & _bcd & ((_mode == 1) & (dataframe['d'] < dataframe['c'])) | (
                    (_mode != 1) & (dataframe['d'] > dataframe['c'])))

    def isGartley(self, dataframe: DataFrame, _mode: int = 1):
        _xab = (dataframe['xab'] >= 0.5) & (dataframe['xab'] <= 0.618)  # 0.618
        _abc = (dataframe['abc'] >= 0.382) & (dataframe['abc'] <= 0.886)
        _bcd = (dataframe['bcd'] >= 1.13) & (dataframe['bcd'] <= 2.618)
        _xad = (dataframe['xad'] >= 0.75) & (dataframe['xad'] <= 0.875)  # 0.786
        return (_xab & _abc & _bcd & _xad & ((_mode == 1) & (dataframe['d'] < dataframe['c'])) | (
                    (_mode != 1) & (dataframe['d'] > dataframe['c'])))

    def isAntiGartley(self, dataframe: DataFrame, _mode: int = 1):
        _xab = (dataframe['xab'] >= 0.500) & (dataframe['xab'] <= 0.886)  # 0.618 -> 0.786
        _abc = (dataframe['abc'] >= 1.000) & (dataframe['abc'] <= 2.618)  # 1.130 -> 2.618
        _bcd = (dataframe['bcd'] >= 1.500) & (dataframe['bcd'] <= 5.000)  # 1.618
        _xad = (dataframe['xad'] >= 1.000) & (dataframe['xad'] <= 5.000)  # 1.272
        return (_xab & _abc & _bcd & _xad & ((_mode == 1) & (dataframe['d'] < dataframe['c'])) | (
                    (_mode != 1) & (dataframe['d'] > dataframe['c'])))

    def isCrab(self, dataframe: DataFrame, _mode: int = 1):
        _xab = (dataframe['xab'] >= 0.500) & (dataframe['xab'] <= 0.875)  # 0.886
        _abc = (dataframe['abc'] >= 0.382) & (dataframe['abc'] <= 0.886)
        _bcd = (dataframe['bcd'] >= 2.000) & (dataframe['bcd'] <= 5.000)  # 3.618
        _xad = (dataframe['xad'] >= 1.382) & (dataframe['xad'] <= 5.000)  # 1.618
        return (_xab & _abc & _bcd & _xad & ((_mode == 1) & (dataframe['d'] < dataframe['c'])) | (
                    (_mode != 1) & (dataframe['d'] > dataframe['c'])))

    def isAntiCrab(self, dataframe: DataFrame, _mode: int = 1):
        _xab = (dataframe['xab'] >= 0.250) & (dataframe['xab'] <= 0.500)  # 0.276 -> 0.446
        _abc = (dataframe['abc'] >= 1.130) & (dataframe['abc'] <= 2.618)  # 1.130 -> 2.618
        _bcd = (dataframe['bcd'] >= 1.618) & (dataframe['bcd'] <= 2.618)  # 1.618 -> 2.618
        _xad = (dataframe['xad'] >= 0.500) & (dataframe['xad'] <= 0.750)  # 0.618
        return (_xab & _abc & _bcd & _xad & ((_mode == 1) & (dataframe['d'] < dataframe['c'])) | (
                    (_mode != 1) & (dataframe['d'] > dataframe['c'])))

    def isShark(self, dataframe: DataFrame, _mode: int = 1):
        _xab = (dataframe['xab'] >= 0.500) & (dataframe['xab'] <= 0.875)  # 0.5 --> 0.886
        _abc = (dataframe['abc'] >= 1.130) & (dataframe['abc'] <= 1.618)  #
        _bcd = (dataframe['bcd'] >= 1.270) & (dataframe['bcd'] <= 2.240)  #
        _xad = (dataframe['xad'] >= 0.886) & (dataframe['xad'] <= 1.130)  # 0.886 --> 1.13
        return (_xab & _abc & _bcd & _xad & ((_mode == 1) & (dataframe['d'] < dataframe['c'])) | (
                    (_mode != 1) & (dataframe['d'] > dataframe['c'])))

    def isAntiShark(self, dataframe: DataFrame, _mode: int = 1):
        _xab = (dataframe['xab'] >= 0.382) & (dataframe['xab'] <= 0.875)  # 0.446 --> 0.618
        _abc = (dataframe['abc'] >= 0.500) & (dataframe['abc'] <= 1.000)  # 0.618 --> 0.886
        _bcd = (dataframe['bcd'] >= 1.250) & (dataframe['bcd'] <= 2.618)  # 1.618 --> 2.618
        _xad = (dataframe['xad'] >= 0.500) & (dataframe['xad'] <= 1.250)  # 1.130 --> 1.130
        return (_xab & _abc & _bcd & _xad & ((_mode == 1) & (dataframe['d'] < dataframe['c'])) | (
                    (_mode != 1) & (dataframe['d'] > dataframe['c'])))

    def is5o(self, dataframe: DataFrame, _mode: int = 1):
        _xab = (dataframe['xab'] >= 1.13) & (dataframe['xab'] <= 1.618)
        _abc = (dataframe['abc'] >= 1.618) & (dataframe['abc'] <= 2.24)
        _bcd = (dataframe['bcd'] >= 0.5) & (dataframe['bcd'] <= 0.625)  # 0.5
        _xad = (dataframe['xad'] >= 0.0) & (dataframe['xad'] <= 0.236)  # negative?
        return (_xab & _abc & _bcd & _xad & ((_mode == 1) & (dataframe['d'] < dataframe['c'])) | (
                    (_mode != 1) & (dataframe['d'] > dataframe['c'])))

    def isWolf(self, dataframe: DataFrame, _mode: int = 1):
        _xab = (dataframe['xab'] >= 1.27) & (dataframe['xab'] <= 1.618)
        _abc = (dataframe['abc'] >= 0) & (dataframe['abc'] <= 5)
        _bcd = (dataframe['bcd'] >= 1.27) & (dataframe['bcd'] <= 1.618)
        _xad = (dataframe['xad'] >= 0.0) & (dataframe['xad'] <= 5)
        return (_xab & _abc & _bcd & _xad & ((_mode == 1) & (dataframe['d'] < dataframe['c'])) | (
                    (_mode != 1) & (dataframe['d'] > dataframe['c'])))

    def isHnS(self, dataframe: DataFrame, _mode: int = 1):
        _xab = (dataframe['xab'] >= 2.0) & (dataframe['xab'] <= 10)
        _abc = (dataframe['abc'] >= 0.90) & (dataframe['abc'] <= 1.1)
        _bcd = (dataframe['bcd'] >= 0.236) & (dataframe['bcd'] <= 0.88)
        _xad = (dataframe['xad'] >= 0.90) & (dataframe['xad'] <= 1.1)
        return (_xab & _abc & _bcd & _xad & ((_mode == 1) & (dataframe['d'] < dataframe['c'])) | (
                    (_mode != 1) & (dataframe['d'] > dataframe['c'])))

    def isConTria(self, dataframe: DataFrame, _mode: int = 1):
        _xab = (dataframe['xab'] >= 0.382) & (dataframe['xab'] <= 0.618)
        _abc = (dataframe['abc'] >= 0.382) & (dataframe['abc'] <= 0.618)
        _bcd = (dataframe['bcd'] >= 0.382) & (dataframe['bcd'] <= 0.618)
        _xad = (dataframe['xad'] >= 0.236) & (dataframe['xad'] <= 0.764)
        return (_xab & _abc & _bcd & _xad & ((_mode == 1) & (dataframe['d'] < dataframe['c'])) | (
                    (_mode != 1) & (dataframe['d'] > dataframe['c'])))

    def isExpTria(self, dataframe: DataFrame, _mode: int = 1):
        _xab = (dataframe['xab'] >= 1.236) & (dataframe['xab'] <= 1.618)
        _abc = (dataframe['abc'] >= 1.000) & (dataframe['abc'] <= 1.618)
        _bcd = (dataframe['bcd'] >= 1.236) & (dataframe['bcd'] <= 2.000)
        _xad = (dataframe['xad'] >= 2.000) & (dataframe['xad'] <= 2.236)
        return (_xab & _abc & _bcd & _xad & ((_mode == 1) & (dataframe['d'] < dataframe['c'])) | (
                    (_mode != 1) & (dataframe['d'] > dataframe['c'])))

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        buy_signal = (
                (self.isABCD(dataframe, 1) | self.isBat(dataframe, 1) | self.isAltBat(dataframe, 1) |
                 self.isButterfly(dataframe, 1) | self.isGartley(dataframe, 1) | self.isCrab(dataframe, 1) |
                 self.isShark(dataframe, 1) | self.is5o(dataframe, 1) | self.isWolf(dataframe, 1) |
                 self.isHnS(dataframe, 1) | self.isConTria(dataframe, 1) | self.isExpTria(dataframe, 1)) &
                (dataframe['close'] <= self.f_last_fib(dataframe, 0.382))
        )
        dataframe.loc[buy_signal, 'enter_long'] = 1

        sell_signal = (
                (self.isABCD(dataframe, -1) | self.isBat(dataframe, -1) | self.isAltBat(dataframe, -1) |
                 self.isButterfly(dataframe, -1) | self.isGartley(dataframe, -1) | self.isCrab(dataframe, -1) |
                 self.isShark(dataframe, -1) | self.is5o(dataframe, -1) | self.isWolf(dataframe, -1) |
                 self.isHnS(dataframe, -1) | self.isConTria(dataframe, -1) | self.isExpTria(dataframe, -1)) &
                (dataframe['close'] >= self.f_last_fib(dataframe, 0.382))
        )
        dataframe.loc[sell_signal, 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_long = (
                (self.isABCD(dataframe, -1) | self.isBat(dataframe, -1) | self.isAltBat(dataframe, -1) |
                 self.isButterfly(dataframe, -1) | self.isGartley(dataframe, -1) | self.isCrab(dataframe, -1) |
                 self.isShark(dataframe, -1) | self.is5o(dataframe, -1) | self.isWolf(dataframe, -1) |
                 self.isHnS(dataframe, -1) | self.isConTria(dataframe, -1) | self.isExpTria(dataframe, -1)) &
                (dataframe['close'] <= self.f_last_fib(dataframe, 0.382))
        )
        dataframe.loc[exit_long, 'exit_long'] = 1

        exit_short = (
                (self.isABCD(dataframe, 1) | self.isBat(dataframe, 1) | self.isAltBat(dataframe, 1) |
                 self.isButterfly(dataframe, 1) | self.isGartley(dataframe, 1) | self.isCrab(dataframe, 1) |
                 self.isShark(dataframe, 1) | self.is5o(dataframe, 1) | self.isWolf(dataframe, 1) |
                 self.isHnS(dataframe, 1) | self.isConTria(dataframe, 1) | self.isExpTria(dataframe, 1)) &
                (dataframe['close'] <= self.f_last_fib(dataframe, -0.618))
        )
        dataframe.loc[exit_short, 'exit_short'] = 1


        return dataframe

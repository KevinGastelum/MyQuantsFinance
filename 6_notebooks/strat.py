# import libs
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# QuantFreedom
from quantfreedom.enums import CandleBodyType
from quantfreedom.helper_funcs import dl_ex_candles, cart_products
from quantfreedom.indicators.tv_indicators import macd_tv, ema_tv
from quantfreedom.strategies.strategy import Strategy

from logging import getLogger
from typing import NamedTuple

# from dotenv import load_dotenv
# load_dotenv()
# api_key = os.getenv('MFX_KEY')
# api_secret = os.getenv('MFX_SECRET')
# Min 31
logger = getLogger('info')

class IndicatorSettingsArrays(NamedTuple):
    ema_length: np.array
    fast_length: np.array
    macd_below: np.array
    signal_smoothing: np.array
    slow_length: np.array

class MACDandEMA(Strategy):
    ema_length = None
    fast_length = None
    macd_below = None
    signal_smoothing = None
    slow_length = None

    def __init__(
        self,
        long_short: str,
        ema_length: np.array,
        fast_length: np.array,
        macd_below: np.array,
        signal_smoothing: np.array,
        slow_length: np.array,
      ):
        self.long_short = long_short
        self.log_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        cart_arrays = cart_product(
            named_tuple=IndicatorSettingsArrays(
                ema_length=ema_length,
                fast_length=fast_length,
                macd_below=macd_below,
                signal_smoothing=signal_smoothing,
                slow_length=slow_length,
            )
        )

        cart_arrays = cart_arrays.T[cart_arrays[1] < cart_arrays[4]].T

        self.indicator_settings_arrays: IndicatorSettingsArrays = IndicatorSettingsArrays(
            ema_length=cart_arrays[0].astype(np.int_),
            fast_length=cart_arrays[1].astype(np.int_),
            macd_below=cart_arrays[2].astype(np.int_),
            signal_smoothing=cart_arrays[3].astype(np.int_),
            slow_length=cart_arrays[4].astype(np.int_),
        )

        



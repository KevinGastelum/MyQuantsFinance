import pandas as pd
import numpy as np
import math
from enum import IntEnum #pip install enum
import mplfinance as mpf #pip install mplfinance
from ta.momentum import rsi as RSI #pip install ta
from ta.volatility import average_true_range as ATR
from ta.trend import ema_indicator as EMA, sma_indicator as SMA
from ta.trend import cci as CCI, adx as ADX, ema_indicator as EMA, sma_indicator as SMA
from sklearn.preprocessing import MinMaxScaler
# from advanced_ta import LorentzianClassification
# https://bitbucket.org/lokiarya/advanced-ta/src/master/advanced_ta/LorentzianClassification/Classifier.py

# TODO add Kernel Regression 

"""
====================
==== Background ====
====================

When using Machine Learning algorithms like K-Nearest Neighbors, choosing an
appropriate distance metric is essential. Euclidean Distance is often used as
the default distance metric, but it may not always be the best choice. This is
because market data is often significantly impacted by proximity to significant
world events such as FOMC Meetings and Black Swan events. These major economic
events can contribute to a warping effect analogous a massive object's
gravitational warping of Space-Time. In financial markets, this warping effect
operates on a continuum, which can analogously be referred to as "Price-Time".

To help to better account for this warping effect, Lorentzian Distance can be
used as an alternative distance metric to Euclidean Distance. The geometry of
Lorentzian Space can be difficult to visualize at first, and one of the best
ways to intuitively understand it is through an example involving 2 feature
dimensions (z=2). For purposes of this example, let's assume these two features
are Relative Strength Index (RSI) and the Average Directional Index (ADX). In
reality, the optimal number of features is in the range of 3-8, but for the sake
of simplicity, we will use only 2 features in this example.

Fundamental Assumptions:
(1) We can calculate RSI and ADX for a given chart.
(2) For simplicity, values for RSI and ADX are assumed to adhere to a Gaussian
    distribution in the range of 0 to 100.
(3) The most recent RSI and ADX value can be considered the origin of a coordinate
    system with ADX on the x-axis and RSI on the y-axis.

Distances in Euclidean Space:
Measuring the Euclidean Distances of historical values with the most recent point
at the origin will yield a distribution that resembles Figure 1 (below).

                       [RSI]
                         |
                         |
                         |
                     ...:::....
               .:.:::••••••:::•::..
             .:•:.:•••::::••::••....::.
            ....:••••:••••••••::••:...:•.
           ...:.::::::•••:::•••:•••::.:•..
           ::•:.:•:•••••••:.:•::::::...:..
 |--------.:•••..•••••••:••:...:::•:•:..:..----------[ADX]
 0        :•:....:•••••::.:::•••::••:.....
          ::....:.:••••••••:•••::••::..:.
           .:...:••:::••••••••::•••....:
             ::....:.....:•::•••:::::..
               ..:..::••..::::..:•:..
                   .::..:::.....:
                         |
                         |
                         |
                         |
                        _|_ 0

       Figure 1: Neighborhood in Euclidean Space

Distances in Lorentzian Space:
However, the same set of historical values measured using Lorentzian Distance will
yield a different distribution that resembles Figure 2 (below).


                        [RSI]
 ::..                     |                    ..:::
  .....                   |                  ......
   .••••::.               |               :••••••.
    .:•••••:.             |            :::••••••.
      .•••••:...          |         .::.••••••.
        .::•••••::..      |       :..••••••..
           .:•••••••::.........::••••••:..
             ..::::••••.•••••••.•••••••:.
               ...:•••••••.•••••••••::.
                 .:..••.••••••.••••..
 |---------------.:•••••••••••••••••.---------------[ADX]
 0             .:•:•••.••••••.•••••••.
             .••••••••••••••••••••••••:.
           .:••••••••••::..::.::••••••••:.
         .::••••••::.     |       .::•••:::.
        .:••••••..        |          :••••••••.
      .:••••:...          |           ..•••••••:.
    ..:••::..             |              :.•••••••.
   .:•....                |               ...::.:••.
  ...:..                  |                   :...:••.
 :::.                     |                       ..::
                         _|_ 0

      Figure 2: Neighborhood in Lorentzian Space


Observations:
(1) In Lorentzian Space, the shortest distance between two points is not
    necessarily a straight line, but rather, a geodesic curve.
(2) The warping effect of Lorentzian distance reduces the overall influence
    of outliers and noise.
(3) Lorentzian Distance becomes increasingly different from Euclidean Distance
    as the number of nearest neighbors used for comparison increases.
"""


# ================= Utils =================

def shift(arr, len, fill_value=0.0):
    return np.pad(arr, (len,), mode='constant', constant_values=(fill_value,))[:arr.size]


def barssince(s: np.array):
    val = np.array([0.0]*s.size)
    c = math.nan
    for i in range(s.size):
        if s[i]: c = 0; continue
        if c >= 0: c += 1
        val[i] = c
    return val


# ================= Types.py =================

# ======================
# ==== Custom Types ====
# ======================

# This section uses PineScript's new Type syntax to define important data structures
# used throughout the script.

class __Config__:
    def __init__(self, **kwargs):
        while kwargs:
            k, v = kwargs.popitem()
            setattr(self, k, v)


class Settings(__Config__):
    source: pd.Series  # Source of the input data
    neighborsCount = 8  # Number of neighbors to consider
    maxBarsBack = 2000  # Maximum number of bars to look back for calculations
    useDynamicExits = False # Dynamic exits attempt to let profits ride by dynamically adjusting the exit threshold based on kernel regression logic

    # EMA Settings
    useEmaFilter = False
    emaPeriod = 200

    # SMA Settings
    useSmaFilter = False
    smaPeriod = 200


class Feature:
    type: str
    param1: int
    param2: int

    def __init__(self, type, param1, param2):
        self.type = type
        self.param1 = param1
        self.param2 = param2


# Nadaraya-Watson Kernel Regression Settings
class KernelFilter(__Config__):
    useKernelSmoothing = False  # Enhance Kernel Smoothing: Uses a crossover based mechanism to smoothen kernel color changes. This often results in less color transitions overall and may result in more ML entry signals being generated.
    lookbackWindow = 8  # Lookback Window: The number of bars used for the estimation. This is a sliding value that represents the most recent historical bars. Recommended range: 3-50
    relativeWeight = 8.0  # Relative Weighting: Relative weighting of time frames. As this value approaches zero, the longer time frames will exert more influence on the estimation. As this value approaches infinity, the behavior of the Rational Quadratic Kernel will become identical to the Gaussian kernel. Recommended range: 0.25-25
    regressionLevel = 25  # Regression Level: Bar index on which to start regression. Controls how tightly fit the kernel estimate is to the data. Smaller values are a tighter fit. Larger values are a looser fit. Recommended range: 2-25
    crossoverLag = 2  # Lag: Lag for crossover detection. Lower values result in earlier crossovers. Recommended range: 1-2


class FilterSettings(__Config__):
    useVolatilityFilter = False,  # Whether to use the volatility filter
    useRegimeFilter = False,  # Whether to use the trend detection filter
    useAdxFilter = False,  # Whether to use the ADX filter
    regimeThreshold = 0.0,  # Threshold for detecting Trending/Ranging markets
    adxThreshold = 0  # Threshold for detecting Trending/Ranging markets

    kernelFilter: KernelFilter


class Filter(__Config__):
    volatility = False
    regime = False
    adx = False


# Label Object: Used for classifying historical data as training data for the ML Model
class Direction(IntEnum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


# ================= KernelFunctions =================

def rationalQuadratic(src: pd.Series, lookback: int, relativeWeight: float, startAtBar: int):
    """
    vectorized calculate for rational quadratic curve
    :param src:
    :param lookback:
    :param relativeWeight:
    :param startAtBar:
    :return:
    """
    currentWeight = [0.0]*len(src)
    cumulativeWeight = 0.0
    for i in range(startAtBar + 2):
        y = src.shift(i, fill_value=0.0)
        w = (1 + (i ** 2 / (lookback ** 2 * 2 * relativeWeight))) ** -relativeWeight
        currentWeight += y.values * w
        cumulativeWeight += w
    val = currentWeight / cumulativeWeight
    val[:startAtBar + 1] = 0.0

    return val


def gaussian(src: pd.Series, lookback: int, startAtBar: int):
    """
    vectorized calculate for gaussian curve
    :param src:
    :param lookback:
    :param startAtBar:
    :return:
    """
    currentWeight = [0.0]*len(src)
    cumulativeWeight = 0.0
    for i in range(startAtBar + 2):
        y = src.shift(i, fill_value=0.0)
        w = math.exp(-(i ** 2) / (2 * lookback ** 2))
        currentWeight += y.values * w
        cumulativeWeight += w
    val = currentWeight / cumulativeWeight
    val[:startAtBar + 1] = 0.0

    return val


#================= MLExtensions =================
# ==========================
# ==== Helper Functions ====
# ==========================

def normalize(src: np.array, range_min=0, range_max=1) -> np.array:
    """
    function Rescales a source value with an unbounded range to a bounded range
    param src: <np.array> The input series
    param range_min: <float> The minimum value of the unbounded range
    param range_max: <float> The maximum value of the unbounded range
    returns <np.array> The normalized series
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return range_min + (range_max - range_min) * scaler.fit_transform(src.reshape(-1,1))[:,0]


def rescale(src: np.array, old_min, old_max, new_min=0, new_max=1) -> np.array:
    """
    function Rescales a source value with a bounded range to anther bounded range
    param src: <np.array> The input series
    param old_min: <float> The minimum value of the range to rescale from
    param old_max: <float> The maximum value of the range to rescale from
    param new_min: <float> The minimum value of the range to rescale to
    param new_max: <float> The maximum value of the range to rescale to 
    returns <np.array> The rescaled series
    """
    rescaled_value = new_min + (new_max - new_min) * (src - old_min) / max(old_max - old_min, 10e-10)
    return rescaled_value


def n_rsi(src: pd.Series, n1, n2) -> np.array:
    """
    function Returns the normalized RSI ideal for use in ML algorithms
    param src: <np.array> The input series
    param n1: <int> The length of the RSI
    param n2: <int> The smoothing length of the RSI
    returns <np.array> The normalized RSI
    """
    return rescale(EMA(RSI(src, n1), n2).values, 0, 100)


def n_cci(highSrc: pd.Series, lowSrc: pd.Series, closeSrc: pd.Series, n1, n2) -> np.array:
    """
    function Returns the normalized CCI ideal for use in ML algorithms
    param highSrc: <np.array> The input series for the high price
    param lowSrc: <np.array> The input series for the low price
    param closeSrc: <np.array> The input series for the close price
    param n1: <int> The length of the CCI
    param n2: <int> The smoothing length of the CCI
    returns <np.array> The normalized CCI
    """
    return normalize(EMA(CCI(highSrc, lowSrc, closeSrc, n1), n2).values)

def n_wt(src: pd.Series, n1=10, n2=11) -> np.array:
    """
    function Returns the normalized WaveTrend Classic series ideal for use in ML algorithms
    param src: <np.array> The input series
    param n1: <int> The first smoothing length for WaveTrend Classic
    param n2: <int> The second smoothing length for the WaveTrend Classic
    returns <np.array> The normalized WaveTrend Classic series
    """
    ema1 = EMA(src, n1)
    ema2 = EMA(abs(src - ema1), n1)
    ci = (src - ema1) / (0.015 * ema2)
    wt1 = EMA(ci, n2)  # tci
    wt2 = SMA(wt1, 4)
    return normalize((wt1 - wt2).values)

def n_adx(highSrc: pd.Series, lowSrc: pd.Series, closeSrc: pd.Series, n1) -> np.array:
    """
    function Returns the normalized ADX ideal for use in ML algorithms
    param highSrc: <np.array> The input series for the high price
    param lowSrc: <np.array> The input series for the low price
    param closeSrc: <np.array> The input series for the close price
    param n1: <int> The length of the ADX
    """
    return rescale(ADX(highSrc, lowSrc, closeSrc, n1).values, 0, 100)
    # TODO: Replicate ADX logic from jdehorty


# =================
# ==== Filters ====
# =================
def regime_filter(src: pd.Series, high: pd.Series, low: pd.Series, useRegimeFilter, threshold) -> np.array:
    """
    regime_filter
    param src: <np.array> The source series
    param high: <np.array> The input series for the high price
    param low: <np.array> The input series for the low price
    param useRegimeFilter: <bool> Whether to use the regime filter
    param threshold: <float> The threshold
    returns <np.array> Boolean indicating whether or not to let the signal pass through the filter
    """
    if not useRegimeFilter: return np.array([True]*len(src))

    # @njit(parallel=True, cache=True)
    def klmf(src: np.array, high: np.array, low: np.array):
        value1 = np.array([0.0]*len(src))
        value2 = np.array([0.0]*len(src))
        klmf = np.array([0.0]*len(src))

        for i in range(len(src)):
            if (high[i] - low[i]) == 0: continue
            value1[i] = 0.2 * (src[i] - src[i - 1 if i >= 1 else 0]) + 0.8 * value1[i - 1 if i >= 1 else 0]
            value2[i] = 0.1 * (high[i] - low[i]) + 0.8 * value2[i - 1 if i >= 1 else 0]

        with np.errstate(divide='ignore',invalid='ignore'):
            omega = np.nan_to_num(np.abs(np.divide(value1, value2)))
        alpha = (-(omega ** 2) + np.sqrt((omega ** 4) + 16 * (omega ** 2))) / 8

        for i in range(len(src)):
            klmf[i] = alpha[i] * src[i] + (1 - alpha[i]) * klmf[i - 1 if i >= 1 else 0]

        return klmf

    filter = np.array([False]*len(src))
    absCurveSlope = np.abs(np.diff(klmf(src.values, high.values, low.values), prepend=0.0))
    exponentialAverageAbsCurveSlope = EMA(pd.Series(absCurveSlope), 200).values
    with np.errstate(divide='ignore',invalid='ignore'):
        normalized_slope_decline = (absCurveSlope - exponentialAverageAbsCurveSlope) / exponentialAverageAbsCurveSlope
    flags = (normalized_slope_decline >= threshold)
    filter[(len(filter) - len(flags)):] = flags
    return filter

def filter_adx(src: pd.Series, high: pd.Series, low: pd.Series, adxThreshold, useAdxFilter, length=14) -> np.array:
    """
    function filter_adx
    param src: <np.array> The source series
    param high: <np.array> The input series for the high price
    param low: <np.array> The input series for the low price
    param adxThreshold: <int> The ADX threshold
    param useAdxFilter: <bool> Whether to use the ADX filter
    param length: <int> The length of the ADX
    returns <np.array> Boolean indicating whether or not to let the signal pass through the filter
    """
    if not useAdxFilter: return np.array([True]*len(src))
    adx = ADX(high, low, src, length).values
    return (adx > adxThreshold)

def filter_volatility(high, low, close, useVolatilityFilter, minLength=1, maxLength=10) -> np.array:
    """
    function filter_volatility
    param high: <np.array> The input series for the high price
    param low: <np.array> The input series for the low price
    param close: <np.array> The input series for the close price
    param useVolatilityFilter: <bool> Whether to use the volatility filter
    param minLength: <int> The minimum length of the ATR
    param maxLength: <int> The maximum length of the ATR
    returns <np.array> Boolean indicating whether or not to let the signal pass through the filter
    """
    if not useVolatilityFilter: return np.array([True]*len(close))
    recentAtr = ATR(high, low, close, minLength).values
    historicalAtr = ATR(high, low, close, maxLength).values
    return (recentAtr > historicalAtr)


# ================ Lorentzian Classifier ================
# INSERT df

class LorentzianClassification:
    
    # from .Types import Feature, Settings, KernelFilter, FilterSettings

    df: pd.DataFrame = None
    features = list[np.array]()
    settings: Settings
    filterSettings: FilterSettings
    # Filter object for filtering the ML predictions
    filter: Filter

    yhat1: np.array
    yhat2: np.array


    # Feature Variables: User-Defined Inputs for calculating Feature Series.
    # Options: ["RSI", "WT", "CCI", "ADX"]
    # FeatureSeries Object: Calculated Feature Series based on Feature Variables
    def series_from(data: pd.DataFrame, feature_string, f_paramA, f_paramB) -> np.array:
        match feature_string:
            case "RSI":
                return n_rsi(data['close'], f_paramA, f_paramB) # ml
            case "WT":
                hlc3 = (data['high'] + data['low'] + data['close']) / 3 #ml
                return n_wt(hlc3, f_paramA, f_paramB)
            case "CCI":
                return n_cci(data['high'], data['low'], data['close'], f_paramA, f_paramB) # ml
            case "ADX":
                return n_adx(data['high'], data['low'], data['close'], f_paramA) #ml


    def __init__(self, data: pd.DataFrame, features: list = None, settings: Settings = None, filterSettings: FilterSettings = None):
        self.df = data.copy()
        self.features = []
        self.filterSettings = None
        self.settings = None
        self.filter = None
        self.yhat1 = None
        self.yhat2 = None

        if features == None:
            features = [
                Feature("RSI", 14, 2),  # f1
                Feature("WT", 10, 11),  # f2
                Feature("CCI", 20, 2),  # f3
                Feature("ADX", 20, 2),  # f4
                Feature("RSI", 9, 2),   # f5
            ]
        if settings == None:
            settings = Settings(source=data['close'])

        if filterSettings == None:
            filterSettings = FilterSettings(
                useVolatilityFilter = True,
                useRegimeFilter = True,
                useAdxFilter = False,
                regimeThreshold=-0.1,
                adxThreshold = 20,
                kernelFilter = KernelFilter()
            )
        if hasattr(filterSettings, 'kernelFilter'):
            self.useKernelFilter = True
        else:
            self.useKernelFilter = False
            filterSettings.kernelFilter = KernelFilter()

        for f in features:
            if type(f) == Feature:
                self.features.append(LorentzianClassification.series_from(data, f.type, f.param1, f.param2))
            else:
                match type(f):
                    case np.ndarray:
                        self.features.append(f)
                    case pd.Series:
                        self.features.append(f.values)
                    case list:
                        self.features.append(np.array(f))

        self.settings = settings
        self.filterSettings = filterSettings
        ohlc4 = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        self.filter = Filter(
            volatility = filter_volatility(data['high'], data['low'], data['close'], filterSettings.useVolatilityFilter, 1, 10), #ml
            regime = regime_filter(ohlc4, data['high'], data['low'], filterSettings.useRegimeFilter, filterSettings.regimeThreshold), #ml
            adx = filter_adx(settings.source, data['high'], data['low'], filterSettings.adxThreshold, filterSettings.useAdxFilter, 14) # .ml
        )
        self.__classify()

    def __classify(self):
        # Derived from General Settings
        maxBarsBackIndex = (len(self.df.index) - self.settings.maxBarsBack) if (len(self.df.index) >= self.settings.maxBarsBack) else 0

        isEmaUptrend = np.where(self.settings.useEmaFilter, (self.df["close"] > EMA(self.df["close"], self.settings.emaPeriod)), True)
        isEmaDowntrend = np.where(self.settings.useEmaFilter, (self.df["close"] < EMA(self.df["close"], self.settings.emaPeriod)), True)
        isSmaUptrend = np.where(self.settings.useSmaFilter, (self.df["close"] > SMA(self.df["close"], self.settings.smaPeriod)), True)
        isSmaDowntrend = np.where(self.settings.useSmaFilter, (self.df["close"] < SMA(self.df["close"], self.settings.smaPeriod)), True)

        """
        =================================
        ==== Next Bar Classification ====
        =================================

        This model specializes specifically in predicting the direction of price action over the course of the next 4 bars. 
        To avoid complications with the ML model, this value is hardcoded to 4 bars but support for other training lengths may be added in the future.

        =========================
        ====  Core ML Logic  ====
        =========================

        Approximate Nearest Neighbors Search with Lorentzian Distance:
        A novel variation of the Nearest Neighbors (NN) search algorithm that ensures a chronologically uniform distribution of neighbors.

        In a traditional KNN-based approach, we would iterate through the entire dataset and calculate the distance between the current bar 
        and every other bar in the dataset and then sort the distances in ascending order. We would then take the first k bars and use their 
        labels to determine the label of the current bar. 

        There are several problems with this traditional KNN approach in the context of real-time calculations involving time series data:
        - It is computationally expensive to iterate through the entire dataset and calculate the distance between every historical bar and
          the current bar.
        - Market time series data is often non-stationary, meaning that the statistical properties of the data change slightly over time.
        - It is possible that the nearest neighbors are not the most informative ones, and the KNN algorithm may return poor results if the
          nearest neighbors are not representative of the majority of the data.

        Previously, the user @capissimo attempted to address some of these issues in several of his PineScript-based KNN implementations by:
        - Using a modified KNN algorithm based on consecutive furthest neighbors to find a set of approximate "nearest" neighbors.
        - Using a sliding window approach to only calculate the distance between the current bar and the most recent n bars in the dataset.

        Of these two approaches, the latter is inherently limited by the fact that it only considers the most recent bars in the overall dataset. 

        The former approach has more potential to leverage historical price action, but is limited by:
        - The possibility of a sudden "max" value throwing off the estimation
        - The possibility of selecting a set of approximate neighbors that are not representative of the majority of the data by oversampling 
          values that are not chronologically distinct enough from one another
        - The possibility of selecting too many "far" neighbors, which may result in a poor estimation of price action

        To address these issues, a novel Approximate Nearest Neighbors (ANN) algorithm is used in this indicator.

        In the below ANN algorithm:
        1. The algorithm iterates through the dataset in chronological order, using the modulo operator to only perform calculations every 4 bars.
           This serves the dual purpose of reducing the computational overhead of the algorithm and ensuring a minimum chronological spacing 
           between the neighbors of at least 4 bars.
        2. A list of the k-similar neighbors is simultaneously maintained in both a predictions array and corresponding distances array.
        3. When the size of the predictions array exceeds the desired number of nearest neighbors specified in settings.neighborsCount, 
           the algorithm removes the first neighbor from the predictions array and the corresponding distance array.
        4. The lastDistance variable is overriden to be a distance in the lower 25% of the array. This step helps to boost overall accuracy 
           by ensuring subsequent newly added distance values increase at a slower rate.
        5. Lorentzian distance is used as a distance metric in order to minimize the effect of outliers and take into account the warping of 
           "price-time" due to proximity to significant economic events.
        """

        src = self.settings.source

        def get_lorentzian_predictions():
            for bar_index in range(maxBarsBackIndex): yield 0

            predictions = []
            distances = []
            y_train_array = np.where(src.shift(4) < src.shift(0), Direction.SHORT, np.where(src.shift(4) > src.shift(0), Direction.LONG, Direction.NEUTRAL))

            class Distances(object):
                batchSize = 50
                lastBatch = 0

                def __init__(self, features):
                    self.size = (len(src) - maxBarsBackIndex)
                    self.features = features
                    self.maxBarsBackIndex = maxBarsBackIndex
                    self.dists = np.array([[0.0] * self.size] * self.batchSize)
                    self.rows = np.array([0.0] * self.batchSize)

                def __getitem__(self, item):
                    batch = math.ceil((item + 1)/self.batchSize) * self.batchSize
                    if batch > self.lastBatch:
                        self.dists.fill(0.0)
                        for feature in self.features:
                            self.rows.fill(0.0)
                            fBatch = feature[(self.maxBarsBackIndex + self.lastBatch):(self.maxBarsBackIndex + batch)]
                            self.rows[:fBatch.size] = fBatch.reshape(-1,)
                            val = np.log(1 + np.abs(self.rows.reshape(-1,1) - feature[:self.size].reshape(1,-1)))
                            self.dists += val
                        self.lastBatch = batch

                    return self.dists[item % self.batchSize]

            dists = Distances(self.features)
            for bar_index in range(maxBarsBackIndex, len(src)):
                lastDistance = -1.0
                span = min(self.settings.maxBarsBack, bar_index + 1)
                for i, d in enumerate(dists[bar_index - maxBarsBackIndex][:span]):
                    if d >= lastDistance and i % 4:
                        lastDistance = d
                        distances.append(d)
                        predictions.append(round(y_train_array[i]))
                        if len(predictions) > self.settings.neighborsCount:
                            lastDistance = distances[round(self.settings.neighborsCount*3/4)]
                            distances.pop(0)
                            predictions.pop(0)
                yield sum(predictions)


        prediction = np.array([p for p in get_lorentzian_predictions()])


        # ============================
        # ==== Prediction Filters ====
        # ============================

        # User Defined Filters: Used for adjusting the frequency of the ML Model's predictions
        filter_all = self.filter.volatility & self.filter.regime & self.filter.adx

        # Filtered Signal: The model's prediction of future price movement direction with user-defined filters applied
        signal = np.where(((prediction > 0) & filter_all), Direction.LONG, np.where(((prediction < 0) & filter_all), Direction.SHORT, None))
        signal[0] = (0 if signal[0] == None else signal[0])
        for i in np.where(signal == None)[0]: signal[i] = signal[i - 1 if i >= 1 else 0]
        
        change = lambda ser, i: (shift(ser, i, fill_value=ser[0]) != shift(ser, i+1, fill_value=ser[0]))

        # Bar-Count Filters: Represents strict filters based on a pre-defined holding period of 4 bars
        barsHeld = []
        isDifferentSignalType = (signal != shift(signal, 1, fill_value=signal[0]))
        _sigFlip = np.where(isDifferentSignalType)[0].tolist()
        if not (len(isDifferentSignalType) in _sigFlip): _sigFlip.append(len(isDifferentSignalType))
        for i, x in enumerate(_sigFlip):
            if i > 0: barsHeld.append(0)
            barsHeld += range(1, x-(-1 if i == 0 else _sigFlip[i-1]))
        isHeldFourBars = (pd.Series(barsHeld) == 4).tolist()
        isHeldLessThanFourBars = (pd.Series(barsHeld) < 4).tolist()

        # Fractal Filters: Derived from relative appearances of signals in a given time series fractal/segment with a default length of 4 bars
        isEarlySignalFlip = (change(signal, 0) & change(signal, 1) & change(signal, 2) & change(signal, 3))
        isBuySignal = ((signal == Direction.LONG) & isEmaUptrend & isSmaUptrend)
        isSellSignal = ((signal == Direction.SHORT) & isEmaDowntrend & isSmaDowntrend)
        isLastSignalBuy = (shift(signal, 4) == Direction.LONG) & shift(isEmaUptrend, 4) & shift(isSmaUptrend, 4)
        isLastSignalSell = (shift(signal, 4) == Direction.SHORT) & shift(isEmaDowntrend, 4) & shift(isSmaDowntrend, 4)
        isNewBuySignal = (isBuySignal & isDifferentSignalType)
        isNewSellSignal = (isSellSignal & isDifferentSignalType)

        crossover   = lambda s1, s2: (s1 > s2) & (shift(s1, 1) < shift(s2, 1))
        crossunder  = lambda s1, s2: (s1 < s2) & (shift(s1, 1) > shift(s2, 1))

        # Kernel Regression Filters: Filters based on Nadaraya-Watson Kernel Regression using the Rational Quadratic Kernel
        # For more information on this technique refer to my other open source indicator located here:
        # https://www.tradingview.com/script/AWNvbPRM-Nadaraya-Watson-Rational-Quadratic-Kernel-Non-Repainting/
        kFilter = self.filterSettings.kernelFilter
        self.yhat1 = rationalQuadratic(src, kFilter.lookbackWindow, kFilter.relativeWeight, kFilter.regressionLevel)
        self.yhat2 = gaussian(src, kFilter.lookbackWindow-kFilter.crossoverLag, kFilter.regressionLevel)
        # Kernel Rates of Change
        wasBearishRate = np.where(shift(self.yhat1, 2) > shift(self.yhat1, 1), True, False)
        wasBullishRate = np.where(shift(self.yhat1, 2) < shift(self.yhat1, 1), True, False)
        isBearishRate = np.where(shift(self.yhat1, 1) > self.yhat1, True, False)
        isBullishRate = np.where(shift(self.yhat1, 1) < self.yhat1, True, False)
        isBearishChange = isBearishRate & wasBullishRate
        isBullishChange = isBullishRate & wasBearishRate
        # Kernel Crossovers
        isBullishCrossAlert = crossover(self.yhat2, self.yhat1)
        isBearishCrossAlert = crossunder(self.yhat2, self.yhat1)
        isBullishSmooth = (self.yhat2 >= self.yhat1)
        isBearishSmooth = (self.yhat2 <= self.yhat1)
        # Kernel Colors
        # plot(kernelEstimate, color=plotColor, linewidth=2, title="Kernel Regression Estimate")
        # Alert Variables
        alertBullish = np.where(kFilter.useKernelSmoothing, isBullishCrossAlert, isBullishChange)
        alertBearish = np.where(kFilter.useKernelSmoothing, isBearishCrossAlert, isBearishChange)
        # Bullish and Bearish Filters based on Kernel
        isBullish = np.where(self.useKernelFilter, np.where(kFilter.useKernelSmoothing, isBullishSmooth, isBullishRate), True)
        isBearish = np.where(self.useKernelFilter, np.where(kFilter.useKernelSmoothing, isBearishSmooth, isBearishRate) , True)

        # ===========================
        # ==== Entries and Exits ====
        # ===========================

        # Entry Conditions: Booleans for ML Model Position Entries
        startLongTrade = isNewBuySignal & isBullish & isEmaUptrend & isSmaUptrend
        startShortTrade = isNewSellSignal & isBearish & isEmaDowntrend & isSmaDowntrend

        # Dynamic Exit Conditions: Booleans for ML Model Position Exits based on Fractal Filters and Kernel Regression Filters
        # lastSignalWasBullish = barssince(startLongTrade) < barssince(startShortTrade)
        # lastSignalWasBearish = barssince(startShortTrade) < barssince(startLongTrade)
        barsSinceRedEntry = barssince(startShortTrade)
        barsSinceRedExit = barssince(alertBullish)
        barsSinceGreenEntry = barssince(startLongTrade)
        barsSinceGreenExit = barssince(alertBearish)
        isValidShortExit = barsSinceRedExit > barsSinceRedEntry
        isValidLongExit = barsSinceGreenExit > barsSinceGreenEntry
        endLongTradeDynamic = isBearishChange & shift(isValidLongExit, 1)
        endShortTradeDynamic = isBullishChange & shift(isValidShortExit, 1)

        # Fixed Exit Conditions: Booleans for ML Model Position Exits based on Bar-Count Filters
        endLongTradeStrict = ((isHeldFourBars & isLastSignalBuy) | (isHeldLessThanFourBars & isNewSellSignal & isLastSignalBuy)) & shift(startLongTrade, 4)
        endShortTradeStrict = ((isHeldFourBars & isLastSignalSell) | (isHeldLessThanFourBars & isNewBuySignal & isLastSignalSell)) & shift(startShortTrade, 4)
        isDynamicExitValid = ~self.settings.useEmaFilter & ~self.settings.useSmaFilter & ~kFilter.useKernelSmoothing
        endLongTrade = self.settings.useDynamicExits & isDynamicExitValid & endLongTradeDynamic | endLongTradeStrict
        endShortTrade = self.settings.useDynamicExits & isDynamicExitValid & endShortTradeDynamic | endShortTradeStrict

        self.df['isEmaUptrend'] = isEmaUptrend
        self.df['isEmaDowntrend'] = isEmaDowntrend
        self.df['isSmaUptrend'] = isSmaUptrend
        self.df['isSmaDowntrend'] = isSmaDowntrend
        self.df["prediction"] = prediction
        self.df["signal"] = signal
        self.df["barsHeld"] = barsHeld
        # self.df["isHeldFourBars"] = isHeldFourBars
        # self.df["isHeldLessThanFourBars"] = isHeldLessThanFourBars
        self.df["isEarlySignalFlip"] = isEarlySignalFlip
        # self.df["isBuySignal"] = isBuySignal
        # self.df["isSellSignal"] = isSellSignal
        self.df["isLastSignalBuy"] = isLastSignalBuy
        self.df["isLastSignalSell"] = isLastSignalSell
        self.df["isNewBuySignal"] = isNewBuySignal
        self.df["isNewSellSignal"] = isNewSellSignal

        self.df["startLongTrade"] = np.where(startLongTrade, self.df['low'], np.NaN)
        self.df["startShortTrade"] = np.where(startShortTrade, self.df['high'], np.NaN)

        self.df["endLongTrade"] = np.where(endLongTrade, self.df['high'], np.NaN)
        self.df["endShortTrade"] = np.where(endShortTrade, self.df['low'], np.NaN)


    # =============================
    # ==== Dump or Return Data ====
    # =============================

    def dump(self, name: str):
        self.df.to_csv(name)


    @property
    def data(self) -> pd.DataFrame:
        return self.df

    # =========================
    # ====    Plotting     ====
    # =========================

    def plot(self, name: str):
        len = self.df.index.size

        # yhat1_g = [self.yhat1[v] if np.where(useKernelSmoothing, isBullishSmooth, isBullishRate)[v] else np.NaN for v in range(self.df.head(len).index.size)]
        # yhat1_r = [self.yhat1[v] if ~np.where(useKernelSmoothing, isBullishSmooth, isBullishRate)[v] else np.NaN for v in range(self.df.head(len).index.size)]
        sub_plots = [
            mpf.make_addplot(self.yhat1.head(len), ylabel="Kernel Regression Estimate", color='blue'),
            mpf.make_addplot(self.yhat2.head(len), ylabel="yhat2", color='gray'),
            mpf.make_addplot(self.df["startLongTrade"], ylabel="startLongTrade", color='green', type='scatter', markersize=120, marker='^'),
            mpf.make_addplot(self.df["startShortTrade"], ylabel="startShortTrade", color='red', type='scatter', markersize=120, marker='v'),
        ]
        s = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'figure.facecolor': 'lightgray'}, edgecolor='black',
                            marketcolors=mpf.make_marketcolors(base_mpf_style='yahoo', inherit=True, alpha=0.2))
        fig, axlist = mpf.plot(self.df[['open', 'high', 'low', 'close']].head(len), type='candle', style=s, addplot=sub_plots, figsize=(30,40) ,returnfig=True)

        for x in range(len):
            y = self.df.loc[self.df.index[x], 'low']
            axlist[0].text(x, y, self.df.loc[self.df.index[x], "prediction"])

        fig.figure.savefig(fname=name)


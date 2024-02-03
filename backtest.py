import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    df = pd.read_csv(filename, parse_dates=True, index_col='Date')
    return df

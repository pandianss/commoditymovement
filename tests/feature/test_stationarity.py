import pytest
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series, p_value_threshold=0.05):
    """
    Helper to check stationarity via ADF.
    """
    result = adfuller(series.dropna())
    return result[1] < p_value_threshold # p-value

def test_feature_stationarity():
    """
    Tier 2: Feature Correctness
    Raw prices are non-stationary, Returns should be stationary.
    """
    # 1. Random Walk (Price) - Non-stationary
    np.random.seed(42)
    rw = np.random.randn(1000).cumsum()
    assert not check_stationarity(pd.Series(rw)), "Random walk should be non-stationary"
    
    # 2. White Noise (Returns) - Stationary
    wn = np.random.randn(1000)
    assert check_stationarity(pd.Series(wn)), "White noise should be stationary"

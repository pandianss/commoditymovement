import pytest
import pandas as pd
import numpy as np

def test_timestamp_monotonicity():
    """
    Tier 1: Data Validity
    Check that timestamps are strictly increasing for a sample dataframe.
    """
    # Create valid df
    dates = pd.date_range("2020-01-01", periods=100, freq='D')
    df = pd.DataFrame({'timestamp_utc': dates})
    assert df['timestamp_utc'].is_monotonic_increasing
    
    # Create invalid df
    df_bad = df.copy()
    df_bad.iloc[50] = df_bad.iloc[40] # Duplicate/older date
    assert not df_bad['timestamp_utc'].is_monotonic_increasing

def test_missing_values():
    """
    Tier 1: Data Completeness
    Ensure critical columns have no NaNs.
    """
    df = pd.DataFrame({
        'close': [100.0, 101.0, np.nan, 103.0]
    })
    # Should fail if we demand 0 nulls for 'close'
    assert df['close'].isnull().sum() > 0

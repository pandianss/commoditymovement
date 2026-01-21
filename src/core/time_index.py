import pandas as pd
from datetime import datetime
from typing import Union

def causal_slice(df: pd.DataFrame, t: Union[datetime, str, pd.Timestamp]) -> pd.DataFrame:
    """
    Returns a slice of the dataframe containing only data with timestamp <= t.
    Ensures that no future data is available to a model or processor at time t.
    """
    ts = pd.to_datetime(t)
    return df[df.index <= ts]

def apply_causal_mask(df: pd.DataFrame, shift_count: int = 1) -> pd.DataFrame:
    """
    Standardizes the temporal shift of features. 
    A shift of 1 means that the feature value at row t is derived from data available at t-1.
    """
    return df.shift(shift_count)

def get_asof_view(df: pd.DataFrame, asof_time: datetime) -> pd.DataFrame:
    """
    Simulates a 'Point-in-Time' view of a dataset.
    """
    return causal_slice(df, asof_time)

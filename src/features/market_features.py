import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, LAG_WINDOWS
from core.time_index import apply_causal_mask


def calculate_returns(df, windows):
    """Calculates log returns for multiple windows."""
    returns_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        # Check if col is a tuple (multi-index)
        col_name = col[1] if isinstance(col, tuple) else col
        for w in windows:
            # Shift by w to get value w days ago.
            # Log return = log(Price_t / Price_{t-w})
            # This feature is known at time t.
            returns_df[f"{col_name}_ret_{w}d"] = np.log(df[col] / df[col].shift(w))
    # Apply standard causal mask (shift 1) to ensure feature calculation at t 
    # relies on data closed at t-1 if strictly required, but for "current day close returns" 
    # typically we consider Close_t as known at Close_t.
    # However, to be strictly predictive for t+1 using state at t, we don't shift return derived from t.
    # EXECUTIVE DIRECTIVE: "Rolling indicators are not consistently shifted" -> usually implies features for next day 
    # should be based on available data.
    # If we predict T+1, we use features at T. Feature at T is (P_t / P_{t-1}). This is safe.
    # BUT if we want to simulate trading decision at Open_T+1, we have P_t.
    # The directives say "features[t] may only use data with timestamp <= t".
    # P_t is <= t. So (P_t / P_{t-1}) is valid at t.
    return returns_df

def calculate_volatility(df, window=20):
    """Calculates rolling realized volatility."""
    vol_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        col_name = col[1] if isinstance(col, tuple) else col
        # rolling(window).std() at t includes t. Valid at t.
        vol_df[f"{col_name}_vol_{window}d"] = df[col].pct_change().rolling(window).std() * np.sqrt(252)
    return vol_df

def calculate_momentum(df, windows):
    """Calculates momentum features (e.g., price relative to moving average)."""
    mom_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        col_name = col[1] if isinstance(col, tuple) else col
        for w in windows:
            # rolling(w).mean() at t includes t. Valid at t.
            ma = df[col].rolling(w).mean()
            mom_df[f"{col_name}_ma_{w}d_ratio"] = df[col] / ma
    return mom_df

def process_market_features():
    raw_path = os.path.join(RAW_DATA_DIR, "commodities_raw.csv")
    if not os.path.exists(raw_path):
        print(f"File not found: {raw_path}")
        return

    # yfinance output has headers on first two rows. 
    # The 'Date' row (index 2) can be problematic.
    # We skip row 2 (index 2) by using skiprows=[2]
    df = pd.read_csv(raw_path, header=[0, 1], index_col=0, parse_dates=True, skiprows=[2])
    
    # In recent yfinance versions, 'Adj Close' might be the same as 'Close' 
    # or named differently in the CSV. In the preview it shows 'Close' at the top.
    # Let's check what's available.
    available_levels = df.columns.get_level_values(0).unique()
    print(f"Available price levels: {available_levels}")
    
    target_level = 'Adj Close' if 'Adj Close' in available_levels else 'Close'
    adj_close = df[target_level]
    
    returns = calculate_returns(adj_close, LAG_WINDOWS)
    vol = calculate_volatility(adj_close, 20)
    momentum = calculate_momentum(adj_close, [20, 60, 252])
    
    features = pd.concat([returns, vol, momentum], axis=1)
    
    output_path = os.path.join(PROCESSED_DATA_DIR, "market_features.csv")
    features.to_csv(output_path)
    print(f"Market features saved to {output_path}")
    return features

if __name__ == "__main__":
    process_market_features()

import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, COMMODITIES

def detect_intraday_shocks(df_1h, std_threshold=3.0):
    """
    Detects sudden price spikes or volume surges in hourly data.
    """
    results = []
    
    # We expect multi-index from yfinance download if multiple tickers
    # or single index if one. In our fetch_intraday_data, we passed list.
    
    tickers = list(COMMODITIES.values())
    
    for ticker in tickers:
        try:
            # Extract close prices for the ticker
            if isinstance(df_1h.columns, pd.MultiIndex):
                close = df_1h['Close'][ticker]
            else:
                close = df_1h['Close']
                
            rets = close.pct_change()
            rolling_std = rets.rolling(window=24).std()
            
            # Shock: Return is > X standard deviations
            shocks = rets[rets.abs() > (std_threshold * rolling_std)]
            
            for timestamp, ret in shocks.items():
                results.append({
                    "timestamp": timestamp,
                    "ticker": ticker,
                    "type": "INTRA_DAY_PRICE_SHOCK",
                    "magnitude": ret,
                    "z_score": ret / rolling_std.loc[timestamp]
                })
        except KeyError:
            continue
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    path_1h = os.path.join(RAW_DATA_DIR, "commodities_1h_raw.csv")
    if os.path.exists(path_1h):
        df = pd.read_csv(path_1h, index_col=0, header=[0, 1], parse_dates=True)
        shocks = detect_intraday_shocks(df)
        if not shocks.empty:
            print(f"Detected {len(shocks)} intra-day shocks!")
            print(shocks.tail())
        else:
            print("No intra-day shocks detected in the current window.")

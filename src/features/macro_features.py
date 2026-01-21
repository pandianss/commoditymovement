import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from core.time_index import apply_causal_mask


def process_macro_features():
    raw_path = os.path.join(RAW_DATA_DIR, "macro_raw.csv")
    if not os.path.exists(raw_path):
        print(f"File not found: {raw_path}")
        return

    # Similar multi-index structure
    df = pd.read_csv(raw_path, header=[0, 1], index_col=0, parse_dates=True, skiprows=[2])
    
    available_levels = df.columns.get_level_values(0).unique()
    target_level = 'Adj Close' if 'Adj Close' in available_levels else 'Close'
    adj_close = df[target_level]
    
    # Macro features:
    # 1. Level of index (normed or log-level)
    # 2. Daily changes
    # 3. Yield curve slope (if we have 10Y and 2Y correctly)
    
    macro_features = pd.DataFrame(index=adj_close.index)
    
    for col in adj_close.columns:
        # Check if col is a tuple (multi-index)
        col_name = col[1] if isinstance(col, tuple) else col
        
        # Log return for macro indices (t vs t-1, valid at t)
        macro_features[f"{col_name}_dret"] = np.log(adj_close[col] / adj_close[col].shift(1))
        
        # 5d smoothing for cleaner signals (rolling mean at t includes t, valid at t)
        macro_features[f"{col_name}_5d_ma"] = adj_close[col].rolling(5).mean()
        
    # Yield Curve Slope (10Y - 2Y)
    # Note: tickers are ^TNX (10Y) and ^IRX (3M, using as proxy for now or check if we got 2Y)
    # In config.py: "UST_10Y": "^TNX", "UST_2Y": "^IRX"
    if "^TNX" in adj_close.columns and "^IRX" in adj_close.columns:
        macro_features["yield_curve_slope"] = adj_close["^TNX"] - adj_close["^IRX"]
        
    output_path = os.path.join(PROCESSED_DATA_DIR, "macro_features.csv")
    macro_features.to_csv(output_path)
    print(f"Macro features saved to {output_path}")
    return macro_features

if __name__ == "__main__":
    process_macro_features()

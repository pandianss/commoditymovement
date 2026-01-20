import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, COMMODITIES

def detect_inflection_points(df, commodity_ticker, std_threshold=2.0):
    """
    Identifies days with abnormal log returns or volatility.
    """
    ret_col = f"{commodity_ticker}_ret_1d"
    vol_col = f"{commodity_ticker}_vol_20d"
    
    if ret_col not in df.columns or vol_col not in df.columns:
        print(f"Required columns for {commodity_ticker} not found.")
        return pd.DataFrame()
        
    returns = df[ret_col]
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    # Thresholds
    upper = mean_ret + (std_threshold * std_ret)
    lower = mean_ret - (std_threshold * std_ret)
    
    # Large Move Detection
    inflections = df[(returns > upper) | (returns < lower)].copy()
    inflections['move_type'] = np.where(inflections[ret_col] > upper, 'POSITIVE_SHOCK', 'NEGATIVE_SHOCK')
    inflections['magnitude'] = inflections[ret_col].abs()
    
    return inflections[['move_type', 'magnitude', ret_col, vol_col]]

def main():
    store_path = os.path.join(PROCESSED_DATA_DIR, "feature_store.csv")
    if not os.path.exists(store_path):
        print("Feature store not found.")
        return
        
    df = pd.read_csv(store_path, index_col=0, parse_dates=True)
    
    all_inflections = []
    
    for name, ticker in COMMODITIES.items():
        print(f"Detecting inflections for {name} ({ticker})...")
        inf = detect_inflection_points(df, ticker)
        if not inf.empty:
            inf['commodity'] = name
            all_inflections.append(inf)
            
    if all_inflections:
        inf_df = pd.concat(all_inflections).sort_index()
        output_path = os.path.join(PROCESSED_DATA_DIR, "inflection_points.csv")
        inf_df.to_csv(output_path)
        print(f"Detected {len(inf_df)} total inflection points across all commodities.")
        print(f"Saved to {output_path}")
        
        # Print top 5 biggest moves
        print("\nTop 5 Inflection Points:")
        print(inf_df.sort_values('magnitude', ascending=False).head(5))

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, COMMODITIES

def calculate_impact_duration(price_series, vol_series, shock_date, threshold_std=1.0, lookback=252, min_recovery_days=5):
    """
    Estimates the impact duration of a shock by detecting when parameters 
    return to their rolling baseline.
    """
    shock_idx = price_series.index.get_loc(shock_date)
    
    # Define Baseline: stats before the shock
    baseline_vol = vol_series.iloc[max(0, shock_idx-lookback):shock_idx].mean()
    baseline_vol_std = vol_series.iloc[max(0, shock_idx-lookback):shock_idx].std()
    
    # Recovery Threshold
    upper_bound = baseline_vol + (threshold_std * baseline_vol_std)
    
    post_shock_vol = vol_series.iloc[shock_idx:]
    
    # Find the first day where vol stays below threshold for min_recovery_days
    recovery_day = None
    consistent_days = 0
    
    for i, vol in enumerate(post_shock_vol):
        if vol <= upper_bound:
            consistent_days += 1
        else:
            consistent_days = 0
            
        if consistent_days >= min_recovery_days:
            recovery_day = post_shock_vol.index[i]
            break
            
    if recovery_day:
        duration = (recovery_day - shock_date).days
        return duration
    return None

def main():
    inf_path = os.path.join(PROCESSED_DATA_DIR, "inflection_points.csv")
    store_path = os.path.join(PROCESSED_DATA_DIR, "feature_store_v2.csv")
    
    if not (os.path.exists(inf_path) and os.path.exists(store_path)):
        print("Required data files not found.")
        return
        
    inf_df = pd.read_csv(inf_path, index_col=0, parse_dates=True)
    store_df = pd.read_csv(store_path, index_col=0, parse_dates=True)
    
    print(f"Analyzing impact duration for {len(inf_df)} shocks...")
    
    durations = []
    
    for idx, row in inf_df.iterrows():
        commodity = row['commodity']
        ticker = COMMODITIES[commodity]
        
        # We use returns and volatility for recovery detection
        # Note: we need the raw series
        ret_col = f"{ticker}_ret_1d"
        vol_col = f"{ticker}_vol_20d"
        
        if ret_col in store_df.columns and vol_col in store_df.columns:
            duration = calculate_impact_duration(
                store_df[ret_col], 
                store_df[vol_col], 
                idx
            )
            durations.append(duration)
        else:
            durations.append(None)
            
    inf_df['impact_duration_days'] = durations
    
    output_path = os.path.join(PROCESSED_DATA_DIR, "inflection_with_impact.csv")
    inf_df.to_csv(output_path)
    
    print(f"Impact analysis complete. Saved to {output_path}")
    print("\nImpact Statistics (Days):")
    print(inf_df['impact_duration_days'].describe())
    
    # Highlight long-duration events
    print("\nTop 5 Long-Persistence Shocks:")
    print(inf_df.sort_values('impact_duration_days', ascending=False).head(5))

if __name__ == "__main__":
    main()

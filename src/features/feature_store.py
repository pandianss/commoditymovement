import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, COMMODITIES

def build_feature_store():
    market_path = os.path.join(PROCESSED_DATA_DIR, "market_features.csv")
    macro_path = os.path.join(PROCESSED_DATA_DIR, "macro_features.csv")
    
    if not (os.path.exists(market_path) and os.path.exists(macro_path)):
        print("Required processed files not found.")
        return

    market = pd.read_csv(market_path, index_col=0, parse_dates=True)
    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    
    # Merge on date
    # We use 'outer' join and then ffill to ensure no holes, but for prediction 
    # we usually want 'inner' to only have days where both exist.
    full_df = pd.concat([market, macro], axis=1).sort_index()
    
    # Forward fill macro data for weekends/holidays if market is open
    full_df = full_df.ffill()
    
    # Create Targets for each commodity (next-day log return)
    # Target: GC=F_ret_1d shifted back 1 day (so row t has return of t+1)
    for commodity, ticker in COMMODITIES.items():
        ret_col = f"{ticker}_ret_1d"
        if ret_col in full_df.columns:
            full_df[f"target_{ticker}_next_ret"] = full_df[ret_col].shift(-1)
            
    # Remove rows with NaN (especially the last row where target is NaN)
    full_df = full_df.dropna()
    
    output_path = os.path.join(PROCESSED_DATA_DIR, "feature_store.csv")
    full_df.to_csv(output_path)
    print(f"Feature store built with {len(full_df.columns)} columns and {len(full_df)} rows.")
    print(f"Saved to {output_path}")
    return full_df

if __name__ == "__main__":
    build_feature_store()

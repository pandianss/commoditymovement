import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR

class PersistenceTrendStrategy:
    """
    Rides the trend following a high-persistence news shock.
    """
    def __init__(self, min_persistence=30):
        self.min_persistence = min_persistence
        
    def generate_signals(self, store_df, inflection_df):
        """
        Generates +1/-1 signals based on inflection points with high persistence.
        """
        signals = pd.DataFrame(0, index=store_df.index, columns=store_df.columns)
        
        # We only care about commodities
        commodities = inflection_df['commodity'].unique()
        
        sig_df = pd.DataFrame(0, index=store_df.index, columns=commodities)
        
        for idx, row in inflection_df.iterrows():
            if row['impact_duration_days'] >= self.min_persistence:
                commodity = row['commodity']
                # Determination of direction
                # POSITIVE_SHOCK -> Long (+1), NEGATIVE_SHOCK -> Short (-1)
                direction = 1 if row['move_type'] == 'POSITIVE_SHOCK' else -1
                
                # Trade duration
                duration = int(row['impact_duration_days'])
                start_date = idx
                
                # Find end date in store_df
                # We stay in the trade for 'duration' days or until next shock
                try:
                    start_pos = store_df.index.get_loc(start_date)
                    end_pos = min(len(store_df) - 1, start_pos + duration)
                    
                    # Fill signals
                    sig_df.iloc[start_pos:end_pos, sig_df.columns.get_loc(commodity)] = direction
                except KeyError:
                    continue
                    
        return sig_df

def run_persistence_demo():
    inf_path = os.path.join(PROCESSED_DATA_DIR, "inflection_with_impact.csv")
    store_path = os.path.join(PROCESSED_DATA_DIR, "feature_store_v2.csv")
    
    if not (os.path.exists(inf_path) and os.path.exists(store_path)):
        print("Required data files not found.")
        return
        
    inf_df = pd.read_csv(inf_path, index_col=0, parse_dates=True)
    store_df = pd.read_csv(store_path, index_col=0, parse_dates=True)
    
    strat = PersistenceTrendStrategy(min_persistence=60) # Only long-term structural shifts
    signals = strat.generate_signals(store_df, inf_df)
    
    return signals, store_df

if __name__ == "__main__":
    sigs, store = run_persistence_demo()
    print("Persistence Strategy Signals Generated.")
    print(sigs.sum())

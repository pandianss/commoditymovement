import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR

class DriftDetector:
    def __init__(self, reference_df):
        self.reference_df = reference_df
        
    def check_drift(self, current_df, feature_cols, p_threshold=0.05):
        """
        Uses KS-test to compare distributions of features between 
        the reference (training) and current (production) data.
        """
        drift_results = {}
        for col in feature_cols:
            if col in self.reference_df.columns and col in current_df.columns:
                stat, p_value = ks_2samp(self.reference_df[col], current_df[col])
                drift_results[col] = {
                    "ks_stat": stat,
                    "p_value": p_value,
                    "drift_detected": p_value < p_threshold
                }
        return drift_results

def main():
    store_path = os.path.join(PROCESSED_DATA_DIR, "feature_store_v2.csv")
    if not os.path.exists(store_path):
        print("Feature store not found.")
        return
        
    df = pd.read_csv(store_path, index_col=0, parse_dates=True)
    
    # Split data to simulate "Reference" (old) and "Current" (new)
    split_date = "2023-01-01"
    reference = df[df.index < split_date]
    current = df[df.index >= split_date]
    
    detector = DriftDetector(reference)
    
    # Check drift for key columns
    target_cols = ['news_sentiment_mean', 'news_volume', 'news_relevance_mean', 'yield_curve_slope']
    results = detector.check_drift(current, target_cols)
    
    print("--- Drift Detection Results ---")
    for col, res in results.items():
        status = "ALERT: DRIFT DETECTED" if res['drift_detected'] else "Stable"
        print(f"{col:25} | P-Value: {res['p_value']:.4f} | {status}")
        
    # Save report
    res_df = pd.DataFrame(results).T
    res_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "drift_report.csv"))

if __name__ == "__main__":
    main()

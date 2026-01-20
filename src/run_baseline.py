import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, WALK_FORWARD_CONFIG
from backtest.walk_forward import run_backtest
from models.baseline import ElasticNetBaseline

def main():
    # Load feature store
    store_path = os.path.join(PROCESSED_DATA_DIR, "feature_store.csv")
    if not os.path.exists(store_path):
        print("Feature store not found. Run feature engineering scripts first.")
        return
        
    df = pd.read_csv(store_path, index_col=0, parse_dates=True)
    
    # Define Target: Gold (GC=F)
    target_col = "target_GC=F_next_ret"
    
    # Feature columns (all except other targets)
    feature_cols = [c for c in df.columns if not c.startswith("target_")]
    
    print(f"Running Baseline Backtest for GOLD...")
    print(f"Number of features: {len(feature_cols)}")
    
    results, preds, actuals = run_backtest(
        ElasticNetBaseline,
        df,
        target_col,
        feature_cols,
        WALK_FORWARD_CONFIG
    )
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(PROCESSED_DATA_DIR, "baseline_results_gold.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()

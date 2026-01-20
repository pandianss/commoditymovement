import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.dataset_spec import DatasetSpec
from contracts.feature_spec import FeatureSpec
from contracts.target_spec import TargetSpec

def main():
    print("--- Data Contracts Demo ---")
    
    # Define Spec
    spec = DatasetSpec(
        dataset_name="Gold_Daily_V1",
        target_col="fwd_ret_5d",
        feature_cols=["close", "rsi_14", "sentiment_score"],
        train_start_date="2020-01-01",
        train_end_date="2020-12-31",
        test_start_date="2021-01-01",
        test_end_date="2021-12-31"
    )
    
    print(f"Defined Spec: {spec.dataset_name}")
    
    # Create Dummy Data
    dates = pd.date_range("2020-01-01", "2021-12-31", freq='D')
    df = pd.DataFrame({
        "timestamp_utc": dates,
        "close": np.random.randn(len(dates)).cumsum() + 1000,
        "rsi_14": np.random.uniform(0, 100, len(dates)),
        "sentiment_score": np.random.uniform(-1, 1, len(dates)),
        "fwd_ret_5d": np.random.normal(0, 0.01, len(dates))
    })
    
    # Validate
    try:
        spec.validate_dataframe(df)
        print("Dataframe passed validation.")
    except Exception as e:
        print(f"Validation failed: {e}")
        
    # Split
    train, test = spec.get_splits(df)
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    # Feature Spec Transform
    f_spec = FeatureSpec(name="momentum", source_col="close", transformation="pct_change")
    df['momentum'] = f_spec.transform(df['close'])
    print("Feature transformation 'momentum' applied.")
    
    # Target Spec
    t_spec = TargetSpec(name="fwd_ret_5d", horizon_days=5)
    print(f"Target defined: {t_spec.name} (Horizon: {t_spec.horizon_days}d)")

if __name__ == "__main__":
    main()

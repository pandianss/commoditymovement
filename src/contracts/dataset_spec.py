from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd

@dataclass
class DatasetSpec:
    """
    Formal specification for a dataset to prevent leakage and ensure reproducibility.
    """
    dataset_name: str
    target_col: str
    feature_cols: List[str]
    time_index_col: str = "timestamp_utc"
    
    # Split Configuration
    train_start_date: str = "2010-01-01"
    train_end_date: str = "2022-12-31"
    test_start_date: str = "2023-01-01"
    test_end_date: str = "2024-12-31"
    
    # Gap between train and test to prevent leakage (e.g., if using lagged features)
    embargo_days: int = 0
    
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validates if the dataframe conforms to the spec.
        """
        required_cols = [self.time_index_col, self.target_col] + self.feature_cols
        missing = [c for c in required_cols if c not in df.columns]
        
        if missing:
            raise ValueError(f"Dataset missing columns: {missing}")
            
        # Check time monotonicity
        if not df[self.time_index_col].is_monotonic_increasing:
             raise ValueError(f"Time index {self.time_index_col} is not monotonic increasing.")
             
        return True

    def get_splits(self, df: pd.DataFrame):
        """
        Returns (train, test) tuple based on the spec.
        """
        # Ensure datetimelike
        df[self.time_index_col] = pd.to_datetime(df[self.time_index_col])
        
        train_mask = (df[self.time_index_col] >= self.train_start_date) & \
                     (df[self.time_index_col] <= self.train_end_date)
                     
        test_mask = (df[self.time_index_col] >= self.test_start_date) & \
                    (df[self.time_index_col] <= self.test_end_date)
                    
        return df[train_mask], df[test_mask]

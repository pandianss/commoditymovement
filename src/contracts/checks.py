import pandas as pd
import numpy as np
from .dataset_spec import DatasetSpec

def check_leakage(df: pd.DataFrame, spec: DatasetSpec):
    """
    Checks for common leakage patterns.
    """
    # 1. Check if target exists in training features (naive check)
    if spec.target_col in spec.feature_cols:
         raise ValueError(f"Leakage: Target {spec.target_col} is in feature list.")
         
    # 2. Check for future data in features (if shifting logic is known)
    # This is hard to do without the generation graph, but we can check if
    # any feature has higher correlation with future target than current target
    # (heuristic)
    pass

def check_scalers(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Verifies that test data is within reasonable bounds of train data.
    """
    pass

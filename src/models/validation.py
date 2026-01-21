import pandas as pd
import numpy as np
from typing import Generator, Tuple

class WalkForwardSplitter:
    """
    Strict Walk-Forward Cross-Validation generator.
    Supports embargo periods to prevent leakage from overlapping labels.
    """
    def __init__(self, n_splits: int = 5, train_window_size: int = None, test_size: int = None, embargo: int = 0):
        self.n_splits = n_splits
        self.train_window_size = train_window_size
        self.test_size = test_size
        self.embargo = embargo

    def split(self, X: pd.DataFrame) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
        """
        Yields (train_indices, test_indices) for each fold.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Simple implementation: Rolling Window
        # If test_size is not provided, we divide the data roughly
        # This logic can be made more sophisticated for 'expanding' vs 'rolling'
        
        if self.test_size is None:
            fold_size = n_samples // (self.n_splits + 1)
        else:
            fold_size = self.test_size
            
        # We start testing after an initial training period
        current_test_start = n_samples - (fold_size * self.n_splits)
        if current_test_start < 0:
             raise ValueError("Data not large enough for requested splits/test_size.")
             
        for _ in range(self.n_splits):
            test_end = current_test_start + fold_size
            
            if test_end > n_samples:
                break
                
            train_end = current_test_start - self.embargo
            
            # Train indices
            if self.train_window_size:
                train_start = max(0, train_end - self.train_window_size)
            else:
                train_start = 0 # Expanding window
                
            train_idx = indices[train_start:train_end]
            test_idx = indices[current_test_start:test_end]
            
            yield train_idx, test_idx
            
            current_test_start += fold_size
            
def purged_kfold(X, n_splits=5, embargo_pct=0.01):
    """
    Utilities for Purged K-Fold (if standard WF is not enough).
    """
    pass

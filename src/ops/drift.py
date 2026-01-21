import numpy as np
import pandas as pd
from typing import Tuple

class DriftDetector:
    """
    Detects statistical drift in model predictions using PSI.
    """
    def __init__(self, baseline_predictions: pd.Series = None):
        self.baseline = baseline_predictions
        
    def calculate_psi(self, baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """
        Calculates Population Stability Index between two distributions.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change (drift detected)
        """
        # Create bins based on baseline
        breakpoints = np.percentile(baseline, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates
        
        if len(breakpoints) < 2:
            return 0.0  # Not enough variation to detect drift
        
        # Calculate distributions
        baseline_counts, _ = np.histogram(baseline, bins=breakpoints)
        current_counts, _ = np.histogram(current, bins=breakpoints)
        
        # Normalize to get percentages
        baseline_pct = baseline_counts / len(baseline)
        current_pct = current_counts / len(current)
        
        # Avoid division by zero
        baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
        current_pct = np.where(current_pct == 0, 0.0001, current_pct)
        
        # PSI formula
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        
        return float(psi)
    
    def check_drift(self, current_predictions: pd.Series, threshold: float = 0.2) -> Tuple[bool, float]:
        """
        Returns (is_drifting, psi_value)
        """
        if self.baseline is None or len(self.baseline) < 10:
            return False, 0.0
        
        if len(current_predictions) < 10:
            return False, 0.0
            
        psi = self.calculate_psi(self.baseline, current_predictions)
        is_drifting = psi >= threshold
        
        return is_drifting, psi
    
    def update_baseline(self, new_baseline: pd.Series):
        """
        Updates the baseline distribution (e.g., after retraining).
        """
        self.baseline = new_baseline

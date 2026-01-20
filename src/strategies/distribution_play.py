import pandas as pd
import numpy as np

class DistributionPlayStrategy:
    """
    Strategy using Quantile Forecasts to hedge or take aggressive positions.
    """
    def __init__(self, upper_q=0.95, lower_q=0.05, threshold=0.02):
        self.upper_q = upper_q
        self.lower_q = lower_q
        self.threshold = threshold
        
    def generate_signals(self, quantile_preds):
        """
        quantile_preds: DataFrame with columns [0.05, 0.5, 0.95]
        """
        signals = pd.Series(0, index=quantile_preds.index)
        
        # Bullish signal: 50th percentile is positive AND 95th percentile is significantly high
        bullish = (quantile_preds[0.5] > 0) & (quantile_preds[self.upper_q] > self.threshold)
        
        # Bearish signal: 50th percentile is negative AND 5th percentile is significantly low
        bearish = (quantile_preds[0.5] < 0) & (quantile_preds[self.lower_q] < -self.threshold)
        
        signals[bullish] = 1
        signals[bearish] = -1
        
        return signals

import pandas as pd
import numpy as np
from typing import Dict, Tuple

class CorrelationDiscovery:
    """
    Research engine for finding lead/lag relationships between signals and returns.
    """
    def __init__(self, max_lag_periods: int = 24):
        self.max_lag_periods = max_lag_periods

    def calculate_cross_correlation(self, signal: pd.Series, target: pd.Series) -> pd.Series:
        """
        Calculates correlation between signal at t-lag and target at t.
        A positive peak at lag > 0 implies signal LEADS target.
        """
        common_idx = signal.index.intersection(target.index)
        s = signal.loc[common_idx]
        t = target.loc[common_idx]
        
        lags = range(-self.max_lag_periods, self.max_lag_periods + 1)
        corrs = {}
        
        for lag in lags:
            # Shift signal: signal.shift(lag) at time T is signal from T-lag
            # Correlation between signal(T-lag) and target(T)
            corrs[lag] = s.shift(lag).corr(t)
            
        return pd.Series(corrs)

    def find_best_lead(self, signal: pd.Series, target: pd.Series) -> Tuple[int, float]:
        """
        Returns (best_lag, max_correlation).
        """
        cc = self.calculate_cross_correlation(signal, target)
        
        # We only care about signals LEADING targets (lag > 0)
        leading_cc = cc[cc.index > 0]
        
        if leading_cc.empty:
            return 0, 0.0
            
        best_lag = leading_cc.abs().idxmax()
        return int(best_lag), float(leading_cc.loc[best_lag])

    def identify_predictive_power(self, signal_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Scans multiple signals against multiple return series to find predictive relationships.
        """
        results = []
        for s_col in signal_df.columns:
            for r_col in returns_df.columns:
                lag, corr = self.find_best_lead(signal_df[s_col], returns_df[r_col])
                results.append({
                    'signal': s_col,
                    'target': r_col,
                    'best_lead_lag': lag,
                    'correlation': corr
                })
        return pd.DataFrame(results).sort_values('correlation', ascending=False)

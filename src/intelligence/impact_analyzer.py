import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ImpactAnalyzer:
    """
    Analyzes the market impact of aligned news events.
    Supporting 'Market Intelligence Core' Requirement A.
    """
    
    def __init__(self, aligned_events: pd.DataFrame):
        """
        :param aligned_events: DataFrame output from EventAligner
        """
        self.events = aligned_events
        
    def compute_conditional_returns(self, group_by=['sentiment'], horizon='fwd_ret_1d'):
        """
        Computes average returns conditional on specific grouping (e.g., sentiment, topic).
        """
        if self.events.empty:
            return pd.DataFrame()
            
        if horizon not in self.events.columns:
            logger.error(f"Horizon {horizon} not found in event data.")
            return pd.DataFrame()
            
        # Group and Aggregate
        stats = self.events.groupby(group_by)[horizon].agg(['mean', 'std', 'count', 'median'])
        
        # Add 'Win Rate' (Probability of positive return)
        stats['win_rate'] = self.events.groupby(group_by)[horizon].apply(lambda x: (x > 0).mean())
        
        return stats
    
    def detect_volatility_shifts(self, pre_col='pre_vol_5d', post_col='post_vol_5d'):
        """
        Compares pre-event volatility vs post-event volatility.
        (Requires EventAligner to populate these cols - future enhancement)
        """
        pass
    
    def compute_shock_probability(self, threshold=0.02, horizon='fwd_ret_1d'):
        """
        Probability that a refined event class leads to a move > threshold.
        """
        if self.events.empty:
            return 0.0
            
        shock_count = self.events[self.events[horizon].abs() > threshold].shape[0]
        total = self.events.shape[0]
        
        return shock_count / total if total > 0 else 0.0

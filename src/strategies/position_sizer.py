import numpy as np
import pandas as pd

class PositionSizer:
    """
    Implements Volatility Targeting and Risk Management logic.
    """
    def __init__(self, target_vol=0.15):
        self.target_vol = target_vol
        
    def calculate_vol_target_weight(self, current_vol):
        """
        Calculates position weight to maintain a target volatility.
        weight = target_vol / current_vol
        """
        if current_vol == 0 or np.isnan(current_vol):
            return 0
        return self.target_vol / current_vol

    def apply_kelly_criterion(self, win_prob, win_loss_ratio):
        """
        Kelly Criterion: f* = p/a - q/b 
        where p is win prob, q is loss prob, a is loss amount, b is win amount.
        Simplified: f* = p - (1-p)/ratio
        """
        if win_loss_ratio == 0:
            return 0
        kelly_f = win_prob - (1 - win_prob) / win_loss_ratio
        return max(0, kelly_f) # No leverage/shorts via Kelly for now

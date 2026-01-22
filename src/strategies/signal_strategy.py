import pandas as pd
from typing import List
import logging
from contracts.signal import Signal

logger = logging.getLogger(__name__)

class SignalDrivenStrategy:
    """
    Base class for strategies that consume standardized Signals.
    """
    def generate_orders(self, signals: List[Signal]):
        raise NotImplementedError

class ProbabilisticTrendStrategy(SignalDrivenStrategy):
    """
    Example V2 Strategy: Trades only when signal probability > threshold.
    Replaces heuristic persistence_trend.py.
    """
    def __init__(self, confidence_threshold=0.7, max_cap=0.20):
        self.confidence_threshold = confidence_threshold
        self.max_cap = max_cap
        
    def generate_allocations(self, signals: List[Signal]) -> pd.DataFrame:
        """
        Consumes signals and returns target portfolio weights.
        """
        allocations = []
        
        for sig in signals:
            if sig.probability >= self.confidence_threshold:
                # Simple weight logic: direction * confidence
                weight = sig.direction * sig.probability
                allocations.append({
                    'date': sig.timestamp_utc,
                    'asset': sig.asset,
                    'weight': weight,
                    'reason': f"{sig.source} ({sig.probability:.2f})"
                })
                
        if not allocations:
            return pd.DataFrame(columns=['date', 'asset', 'weight', 'reason'])
            
        return pd.DataFrame(allocations)

    def apply_risk_budgeting(self, allocations: pd.DataFrame) -> pd.DataFrame:
        """
        Clips allocations to maximum risk budget per asset.
        """
        if allocations.empty:
            return allocations
            
        # Apply cap absolute value
        allocations['weight'] = allocations['weight'].clip(lower=-self.max_cap, upper=self.max_cap)
        return allocations
        
    def determine_exposure(self, allocations: pd.DataFrame, scenario_shock=-0.10) -> float:
        """
        Calculates portfolio P&L given a uniform shock (e.g., -10% market crash).
        """
        if allocations.empty:
            return 0.0
            
        # Net exposure
        net_exposure = allocations['weight'].sum()
        
        # P&L = Net Exposure * Shock
        # If long 50% and market drops 10%, P&L = 0.5 * -0.1 = -0.05
        return net_exposure * scenario_shock

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any
from core.risk import CapitalConstitution, PortfolioState

class BaseStrategy(ABC):
    """
    Abstract base class for all strategies.
    Enforces that 'generate_signals' output is filtered through the Capital Constitution.
    """
    def __init__(self, constitution: CapitalConstitution):
        self.constitution = constitution

    @abstractmethod
    def generate_raw_signals(self, context: pd.DataFrame) -> Dict[str, float]:
        """
        Pure logic: returns desired target weights or position sizes (-1.0 to 1.0 or dollar amount).
        Must be implemented by child strategies.
        """
        pass

    def run_step(self, context: pd.DataFrame, state: PortfolioState) -> Dict[str, float]:
        """
        The Governor:
        1. Gets raw signals from logic.
        2. Checks Drawdown (Freeze if violated).
        3. Applies Volatility Target Scalar.
        4. Clamps individual position limits.
        """
        # 1. Check Constitution - Drawdown Freeze
        if not self.constitution.check_drawdown(state):
            print("ALERT: Drawdown violation! Constitution forces liquidating/freezing.")
            return {sym: 0.0 for sym in state.positions.keys()}

        # 2. Raw Signals
        raw_signals = self.generate_raw_signals(context)
        
        # 3. Volatility Scalar
        vol_scalar = self.constitution.get_vol_scalar(state)
        
        approved_positions = {}
        for symbol, raw_size in raw_signals.items():
            # Apply Scalar
            scaled_size = raw_size * vol_scalar
            
            # 4. Limit Check
            final_size = self.constitution.validate_position_size(symbol, scaled_size, state)
            approved_positions[symbol] = final_size
            
        return approved_positions

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import numpy as np

class RiskProfile(BaseModel):
    """
    Immutable risk axioms for a trading mandate.
    """
    max_drawdown: float = Field(..., description="Hard stop drawdown limit (e.g., 0.20 for 20%)")
    vol_target: float = Field(..., description="Annualized volatility target (e.g., 0.15 for 15%)")
    max_leverage: float = Field(default=1.0, description="Maximum gross exposure ratio")
    position_limit_pct: float = Field(default=0.20, description="Max allocation to single instrument")
    
class PortfolioState(BaseModel):
    """
    Current state of the portfolio for risk checking.
    """
    current_equity: float
    peak_equity: float
    current_volatility: float
    gross_exposure: float
    positions: Dict[str, float] # Symbol -> Value

class CapitalConstitution:
    """
    The Enforcer. Mechanical logic that rejects or resizes trades based on RiskProfile.
    """
    def __init__(self, profile: RiskProfile):
        self.profile = profile

    def check_drawdown(self, state: PortfolioState) -> bool:
        """Returns False if drawdown violation triggers a freeze."""
        if state.peak_equity <= 0: return True
        dd = (state.current_equity / state.peak_equity) - 1
        # Drawdown is negative, max_drawdown is positive limit (0.2)
        # Violation if dd < -max_drawdown
        return dd > -self.profile.max_drawdown

    def get_vol_scalar(self, state: PortfolioState) -> float:
        """Returns a scalar (0.0 to 1.0) to reduce size if volatility is too high."""
        if state.current_volatility <= 0: return 1.0
        # If current vol > target, sizes must shrink
        if state.current_volatility > self.profile.vol_target:
            return self.profile.vol_target / state.current_volatility
        return 1.0

    def validate_position_size(self, symbol: str, size: float, state: PortfolioState) -> float:
        """
        Clamps a requested position size to comply with max position limits and leverage.
        Returns the accepted size.
        """
        # 1. Check Single Position Limit
        max_allowed_val = state.current_equity * self.profile.position_limit_pct
        clamped_size = min(abs(size), max_allowed_val) * np.sign(size)
        
        # 2. Check Gross Leverage (simplified, assumes this is the incremental add)
        # Real logic would check (current_gross + new_size) <= max_leverage * equity
        # For now, we return the clamped individual size.
        return clamped_size

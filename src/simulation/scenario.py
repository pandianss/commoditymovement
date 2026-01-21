import numpy as np
import pandas as pd
from typing import List, Dict
from core.contracts import PredictionArtifact
from core.risk import RiskProfile, CapitalConstitution, PortfolioState

class ScenarioEngine:
    """
    Simulates PnL paths considering prediction uncertainty and news shocks.
    """
    def __init__(self, risk_profile: RiskProfile):
        self.constitution = CapitalConstitution(risk_profile)

    def run_monte_carlo(self, 
                        initial_equity: float, 
                        predictions: List[PredictionArtifact], 
                        num_sims: int = 1000) -> pd.DataFrame:
        """
        Generates N equity curves based on prediction distributions.
        """
        paths = np.zeros((len(predictions), num_sims))
        paths[0, :] = initial_equity
        
        # Simple simulation: assume daily rebalancing based on prediction signal
        # Logic: Position = pred.median * scaler (naive strategy for sim)
        
        current_equities = np.full(num_sims, initial_equity)
        
        for t, pred in enumerate(predictions[:-1]):
            # 1. Generate random return from prediction distribution (Normal assumption approx)
            # sigma approximated from Conf Interval 90% (1.645 sigma on each side)
            # width = upper - lower = 2 * 1.645 * sigma -> sigma = width / 3.29
            width = pred.upper_bound - pred.lower_bound
            sigma = width / 3.29 if width > 0 else 0.01
            
            sim_returns = np.random.normal(pred.median, sigma, num_sims)
            
            # 2. Position Sizing Logic (Simulated Base Strategy)
            # Assume we target 100% allocation if median return positive, -100% if negative (Binary)
            # Then clamped by Constitution
            
            for i in range(num_sims):
                # Simulated 'State' for this path
                # Note: Keeping full state per sim path is expensive, simplified here
                state = PortfolioState(
                    current_equity=current_equities[i],
                    peak_equity=max(current_equities[i], initial_equity), # simplified peak
                    current_volatility=0.15, # Constant vol assumption for demo
                    gross_exposure=0.0,
                    positions={}
                )
                
                # Check Risk Drawdown
                if not self.constitution.check_drawdown(state):
                    # Frozen
                    ret = 0.0
                else:
                    # Strategy Logic
                    raw_signal = 1.0 if pred.median > 0 else -0.5
                    position = self.constitution.validate_position_size("SIM", raw_signal * state.current_equity, state)
                    
                    # PnL
                    # position is Amt in dollars. return is %
                    pnl = position * sim_returns[i]
                    current_equities[i] += pnl
            
            paths[t+1, :] = current_equities
            
        return pd.DataFrame(paths, index=[p.timestamp for p in predictions])

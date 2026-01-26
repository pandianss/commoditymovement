import pandas as pd
import numpy as np

class MacroRegimeClassifier:
    """
    Identifies macro market regimes, specifically detecting 'Crisis' scenarios
    based on volatility, drawdowns, and correlated asset moves.
    """
    def __init__(self, macro_df: pd.DataFrame):
        self.df = macro_df

    def detect_crisis(self, sp500_col="^GSPC_Close", vix_col="^VIX_Close"):
        """
        Detects if the current macro environment is in a 'CRISIS' state.
        Indicators:
        - S&P 500 Drawdown > 15% (Correction/Crisis)
        - VIX > 30 (Extreme Fear)
        - DXY/Gold Correlation (optional)
        """
        if sp500_col not in self.df.columns or vix_col not in self.df.columns:
            return "STABLE", 0.0

        # 1. Calc S&P Drawdown
        sp500 = self.df[sp500_col]
        rolling_max = sp500.expanding().max()
        drawdown = (sp500 / rolling_max) - 1
        current_dd = drawdown.iloc[-1]

        # 2. VIX Level
        vix = self.df[vix_col]
        current_vix = vix.iloc[-1]

        # Classification Logic
        if current_dd < -0.20 or current_vix > 35:
            return "CRISIS", current_dd
        elif current_dd < -0.10 or current_vix > 25:
            return "VOLATILE", current_dd
        else:
            return "STABLE", current_dd

    def get_failsafe_recommendation(self, current_regime):
        """
        Pivots strategy based on the identified regime.
        """
        if current_regime == "CRISIS":
            return ["GOLD", "GOLD_BEES", "LIQUID_BEES", "SUNPHARMA"], "DEFENSIVE_PIVOT"
        elif current_regime == "VOLATILE":
            return ["GOLD", "HUL", "POWERGRID"], "CAUTIOUS"
        else:
            return None, "OPPORTUNISTIC"

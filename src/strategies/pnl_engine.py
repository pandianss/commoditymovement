import pandas as pd
import numpy as np

class PnLEngine:
    """
    Core engine for calculating P&L and performance metrics.
    """
    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital
        
    def calculate_metrics(self, daily_returns):
        """
        Calculates Sharpe, Drawdown, etc. from a series of strategy returns.
        """
        if daily_returns.empty:
            return {}
            
        cum_returns = (1 + daily_returns).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns / peak) - 1
        
        sharpie = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
        
        metrics = {
            "Total Return": cum_returns.iloc[-1] - 1,
            "Annualized Return": (cum_returns.iloc[-1] ** (252 / len(daily_returns))) - 1 if len(daily_returns) > 0 else 0,
            "Annualized Vol": daily_returns.std() * np.sqrt(252),
            "Sharpe Ratio": sharpie,
            "Max Drawdown": drawdown.min(),
            "Profit Factor": daily_returns[daily_returns > 0].sum() / abs(daily_returns[daily_returns < 0].sum()) if daily_returns[daily_returns < 0].sum() != 0 else np.inf
        }
        return metrics

    def backtest(self, signals, price_returns):
        """
        Computes strategy P&L based on signals (shifted by 1 day to avoid lookahead).
        signals: pd.Series where 1 is Long, -1 is Short, 0 is Flat.
        price_returns: pd.Series of daily log or simple returns.
        """
        # Signals should be applied to the NEXT day's return
        strategy_returns = signals.shift(1).fillna(0) * price_returns
        return strategy_returns

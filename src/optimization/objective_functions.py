import numpy as np
import pandas as pd

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculates the annualized Sharpe Ratio.
    Assumes daily returns.
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate/252
    annualized_return = excess_returns.mean() * 252
    annualized_vol = returns.std() * np.sqrt(252)
    
    return annualized_return / annualized_vol

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, target_return: float = 0.0) -> float:
    """
    Calculates the annualized Sortino Ratio.
    Only penalizes downside volatility.
    """
    if returns.empty:
        return 0.0
        
    downside_returns = returns[returns < target_return]
    if downside_returns.empty or downside_returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate/252
    annualized_return = excess_returns.mean() * 252
    downside_vol = downside_returns.std() * np.sqrt(252)
    
    return annualized_return / downside_vol

def calculate_calmar_ratio(returns: pd.Series, window: int = 252) -> float:
    """
    Calculates the Calmar Ratio (Annualized Return / Max Drawdown).
    """
    if returns.empty:
        return 0.0
        
    # Calculate Max Drawdown
    cumulative = (1 + returns).cumprod()
    peak = cumulative.rolling(window=window, min_periods=1).max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = abs(drawdown.min())
    
    if max_drawdown == 0:
        return 0.0
        
    annualized_return = returns.mean() * 252
    return annualized_return / max_drawdown

def calculate_profit_factor(returns: pd.Series) -> float:
    """
    Calculates Profit Factor (Gross Profit / Gross Loss).
    """
    if returns.empty:
        return 0.0
        
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
        
    return gross_profit / gross_loss

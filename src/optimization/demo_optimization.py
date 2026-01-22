import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import logging
from src.optimization.wfo import WalkForwardOptimizer
from src.strategies.signal_strategy import ProbabilisticTrendStrategy
from src.contracts.signal import Signal

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo_opt")

def create_mock_data():
    """
    Creates synthetic price and signal data for testing.
    """
    dates = pd.date_range(start="2024-01-01", end="2025-06-01", freq="1h")
    prices = pd.DataFrame(index=dates)
    prices['GOLD'] = 2000 + np.cumsum(np.random.randn(len(dates)))
    
    # Mock Signals
    # Create a signal dataframe
    signals_data = []
    for i in range(0, len(dates), 24): # Daily signals
        dt = dates[i]
        
        # Random signal
        direction = 1.0 if np.random.random() > 0.5 else -1.0
        prob = 0.5 + (np.random.random() * 0.4) # 0.5 to 0.9
        
        signals_data.append({
            'timestamp_utc': dt,
            'asset': 'GOLD',
            'direction': direction,
            'probability': prob,
            'source': 'Mock_TCN'
        })
        
    signals_df = pd.DataFrame(signals_data)
    signals_df['timestamp_utc'] = pd.to_datetime(signals_df['timestamp_utc'])
    signals_df.set_index('timestamp_utc', drop=False, inplace=True)
    
    return prices, signals_df

def param_space(trial):
    """
    Define Hyperparameter Space for ProbabilisticTrendStrategy.
    """
    return {
        'confidence_threshold': trial.suggest_float('confidence_threshold', 0.55, 0.85, step=0.05),
        'max_cap': trial.suggest_float('max_cap', 0.10, 0.50, step=0.10)
    }

def run_demo():
    print("Generating Mock Data...")
    prices, signals = create_mock_data()
    print(f"Prices: {prices.shape}, Signals: {signals.shape}")
    
    print("\n--- Starting Walk-Forward Optimization ---")
    wfo = WalkForwardOptimizer(
        strategy_class=ProbabilisticTrendStrategy,
        data=prices,
        signals=signals,
        train_window_days=180, # Short windows for demo speed
        test_window_days=30,
        step_days=30
    )
    
    results = wfo.run_wfo(param_space, n_trials=5) # 5 trials per fold for speed
    
    print("\n--- Optimization Complete ---")
    print(results)
    
    print("\nMean OOS Sharpe:", results['oos_sharpe'].mean())

if __name__ == "__main__":
    run_demo()

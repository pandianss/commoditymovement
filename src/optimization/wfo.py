import pandas as pd
import numpy as np
from typing import Dict, Any, Type, Callable, List
from src.backtest.walk_forward import WalkForwardSplitter
from src.optimization.optimizer import StrategyOptimizer
import logging

logger = logging.getLogger("wfo")

class WalkForwardOptimizer:
    def __init__(self, 
                 strategy_class: Type, 
                 data: pd.DataFrame, 
                 signals: pd.DataFrame,
                 train_window_days: int = 365,
                 test_window_days: int = 90,
                 step_days: int = 30):
        
        self.strategy_class = strategy_class
        self.data = data
        self.signals = signals
        self.splitter = WalkForwardSplitter(
            train_window=train_window_days,
            test_window=test_window_days,
            step=step_days
        )
        
    def run_wfo(self, 
                param_space: Callable[[Any], Dict[str, Any]], 
                n_trials: int = 20) -> pd.DataFrame:
        
        splits = list(self.splitter.split(self.data))
        results = []
        
        for i, (train_data, test_data) in enumerate(splits):
            logger.info(f"--- WFO Fold {i+1}/{len(splits)} ---")
            logger.info(f"Train: {train_data.index[0]} to {train_data.index[-1]}")
            logger.info(f"Test: {test_data.index[0]} to {test_data.index[-1]}")
            
            # 1. OPTIMIZE on TRAIN
            # Filter signals for training period
            train_signals = self.signals.loc[self.signals.index.isin(train_data.index)]
            
            optimizer = StrategyOptimizer(self.strategy_class, train_data, train_signals)
            best_params = optimizer.optimize(param_space, n_trials=n_trials)
            
            # 2. VALIDATE on TEST (OOS)
            test_signals = self.signals.loc[self.signals.index.isin(test_data.index)]
            
            # Run Backtest on Test Data with Best Params
            # Note: We need to import BacktestEngine locally to avoid circular issues if any
            from src.backtest.engine import BacktestEngine
            from src.optimization.objective_functions import calculate_sharpe_ratio
            from src.contracts.signal import Signal
            
            # Convert signals to objects (Optimization: Assuming signals are needed for allocation)
            test_signal_objects = []
            for _, row in test_signals.iterrows():
                test_signal_objects.append(Signal(
                    timestamp_utc=row.get('timestamp_utc', row.name),
                    asset=row['asset'],
                    signal_type=row.get('signal_type', 'PREDICTION'),
                    direction=row['direction'],
                    probability=row['probability'],
                    horizon=row.get('horizon', '1d'),
                    source=row.get('source', 'Optimizer')
                ))
            
            strategy = self.strategy_class(**best_params)
            allocations = strategy.generate_allocations(test_signal_objects)
            
            engine = BacktestEngine(initial_capital=100_000, slippage_bps=0, commission_bps=0)
            engine.run_backtest(test_data, allocations)
            
            # History is a list of PortfolioState objects
            history_data = [vars(s) for s in engine.history]
            perf = pd.DataFrame(history_data)
            if perf.empty:
                returns = pd.Series()
            else:
                returns = perf.set_index('date')['equity'].pct_change().dropna()
            
            oos_sharpe = calculate_sharpe_ratio(returns)
            oos_return = returns.sum()
            
            results.append({
                'fold': i,
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'oos_sharpe': oos_sharpe,
                'oos_return': oos_return,
                'best_params': best_params
            })
            
        return pd.DataFrame(results)

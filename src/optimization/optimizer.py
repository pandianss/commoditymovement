import optuna
import pandas as pd
import logging
from typing import Dict, Any, Callable, Type, List
from src.backtest.engine import BacktestEngine
from src.optimization.objective_functions import calculate_sharpe_ratio, calculate_calmar_ratio

logger = logging.getLogger("optimizer")

class StrategyOptimizer:
    def __init__(self, 
                 strategy_class: Type, 
                 data: pd.DataFrame, 
                 signals: pd.DataFrame,
                 base_params: Dict[str, Any] = None):
        """
        Generic Optimizer for Strategies.
        """
        self.strategy_class = strategy_class
        self.data = data
        self.signals = signals
        self.base_params = base_params if base_params else {}
        
        # Pre-convert signals to objects for speed
        from src.contracts.signal import Signal
        self.signal_objects = []
        if not self.signals.empty:
            # Vectorized approach or simple iteration
            for _, row in self.signals.iterrows():
                self.signal_objects.append(Signal(
                    timestamp_utc=row.get('timestamp_utc', row.name), # Handle index if needed
                    asset=row['asset'],
                    signal_type=row.get('signal_type', 'PREDICTION'),
                    direction=row['direction'],
                    probability=row['probability'],
                    horizon=row.get('horizon', '1d'),
                    source=row.get('source', 'Optimizer')
                ))
        
    def optimize(self, 
                 param_space: Callable[[optuna.Trial], Dict[str, Any]], 
                 n_trials: int = 50, 
                 metric: str = 'sharpe') -> Dict[str, Any]:
        """
        Run Optuna optimization.
        param_space: Function that takes a trial and returns a dictionary of sampled params.
        """
        
        def objective(trial):
            # 1. Sample Hyperparameters
            sampled_params = param_space(trial)
            full_params = {**self.base_params, **sampled_params}
            
            # 2. Instantiate Strategy
            strategy = self.strategy_class(**full_params)
            
            # 3. generate Allocations
            
            # 3. generate Allocations
            # Convert signal DF to objects
            # Assuming signals DF has columns: timestamp_utc, asset, direction, probability, source
            signal_objects = []
            for _, row in self.signals.iterrows():
                # We need to ensure the row has all required fields or defaults
                # This is a bit slow for optimization, but correct for architecture.
                # Optimization Tip: Move this conversion OUTSIDE the objective loop if signals don't change.
                pass 
                
            # Actually, to make this fast, we should convert signals ONCE in __init__.
            
            allocations = strategy.generate_allocations(self.signal_objects)
             
            # 4. Run Backtest
            # Disable slippage/costs for optimization speed if desired, or keep reasonable defaults.
            # Using 0 bps for "fast/ideal" optimization check.
            engine = BacktestEngine(initial_capital=100_000, slippage_bps=0, commission_bps=0)
            
            try:
                engine.run_backtest(self.data, allocations)
                performance = pd.DataFrame([vars(s) for s in engine.history])
                
                # 5. Calculate Objective
                # Performance df has 'equity' column
                if performance.empty:
                    return float('-inf')
                    
                returns = performance.set_index('date')['equity'].pct_change().dropna()
                
                if metric == 'sharpe':
                    score = calculate_sharpe_ratio(returns)
                elif metric == 'calmar':
                    score = calculate_calmar_ratio(returns)
                else:
                    score = returns.sum()
                    
                return score
                
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return float('-inf')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best params: {study.best_params}")
        logger.info(f"Best value: {study.best_value}")
        
        return study.best_params

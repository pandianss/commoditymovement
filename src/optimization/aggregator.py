import pandas as pd
import logging

logger = logging.getLogger("aggregator")

class MetricsAggregator:
    """
    Aggregates results from Walk-Forward Optimization folds.
    """
    
    @staticmethod
    def aggregate(wfo_results: pd.DataFrame) -> pd.DataFrame:
        """
        Computes summary statistics across folds.
        """
        if wfo_results.empty:
            return pd.DataFrame()
            
        summary = {
            'mean_oos_sharpe': wfo_results['oos_sharpe'].mean(),
            'median_oos_sharpe': wfo_results['oos_sharpe'].median(),
            'min_oos_sharpe': wfo_results['oos_sharpe'].min(),
            'mean_oos_return': wfo_results['oos_return'].mean(),
            'total_oos_return': wfo_results['oos_return'].sum(),
            'robustness_score': wfo_results['oos_sharpe'].mean() / (wfo_results['oos_sharpe'].std() + 1e-6)
        }
        
        return pd.DataFrame([summary])
        
    @staticmethod
    def get_best_params_across_folds(wfo_results: pd.DataFrame) -> dict:
        """
        Extracts the most stable parameters or the latest fold's params.
        For now, returns last fold's params as 'current best'.
        """
        if wfo_results.empty:
            return {}
            
        last_fold = wfo_results.iloc[-1]
        return last_fold['best_params']

import pandas as pd
import numpy as np

class EnsembleDistribution:
    """
    Ensembles multiple quantile predictions (tabular + sequence models).
    """
    def __init__(self, weights={'lgbm': 0.5, 'tcn': 0.5}):
        self.weights = weights
        
    def ensemble(self, lgbm_preds, tcn_preds):
        """
        Expects DataFrames with the same index and quantile columns.
        """
        if not (lgbm_preds.index.equals(tcn_preds.index)):
            # Fallback to inner join if indices differ
            common_idx = lgbm_preds.index.intersection(tcn_preds.index)
            lgbm_preds = lgbm_preds.loc[common_idx]
            tcn_preds = tcn_preds.loc[common_idx]
            
        final_preds = (lgbm_preds * self.weights['lgbm']) + (tcn_preds * self.weights['tcn'])
        return final_preds

def evaluate_ensemble(actuals, ensemble_preds):
    """
    Calculates Pinball Loss for quantile evaluation.
    """
    from sklearn.metrics import mean_squared_error
    
    results = {}
    for q in ensemble_preds.columns:
        pred = ensemble_preds[q]
        error = actuals - pred
        pinball = np.mean(np.maximum(q * error, (q - 1) * error))
        results[f"pinball_q{q}"] = pinball
        
    results["mse"] = mean_squared_error(actuals, ensemble_preds[0.5])
    return results

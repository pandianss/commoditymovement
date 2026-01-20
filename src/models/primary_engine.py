import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import sys

class QuantileGBM:
    """
    LightGBM model for multiple quantile regression.
    """
    def __init__(self, quantiles=[0.05, 0.5, 0.95], params=None):
        self.quantiles = quantiles
        self.models = {}
        self.params = params or {
            'objective': 'quantile',
            'metric': 'quantile',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
    def fit(self, X, y):
        for q in self.quantiles:
            # print(f"Training model for quantile {q}...")
            model = lgb.LGBMRegressor(alpha=q, **self.params)
            model.fit(X, y)
            self.models[q] = model
            
    def predict(self, X):
        preds = {}
        for q, model in self.models.items():
            preds[q] = model.predict(X)
        return pd.DataFrame(preds)

def get_quantile_model():
    return QuantileGBM

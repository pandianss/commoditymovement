import pandas as pd
import numpy as np
import os
import sys

class MonteCarloSimulator:
    def __init__(self, start_price, model, last_features, steps=30, paths=1000):
        self.start_price = start_price
        self.model = model
        self.last_features = last_features
        self.steps = steps
        self.paths = paths
        
    def simulate(self):
        """
        Generates simulated price paths.
        Simplified version: assumes the predicted return distribution 
        remains stable or evolves based on price.
        """
        all_paths = np.zeros((self.paths, self.steps))
        all_paths[:, 0] = self.start_price
        
        # Get return distribution for next step
        # Note: model.predict returns percentiles [0.05, 0.5, 0.95]
        dist = self.model.predict(self.last_features)
        
        q05 = dist.iloc[0, 0]
        q50 = dist.iloc[0, 1]
        q95 = dist.iloc[0, 2]
        
        # Estimate mean and std from quantiles (assuming normal-ish for simplicity)
        mu = q50
        sigma = (q95 - q05) / (2 * 1.645) # 1.645 is z-score for 90% range
        
        for p in range(self.paths):
            current_price = self.start_price
            for t in range(1, self.steps):
                # Daily log return sampled from the estimated distribution
                ret = np.random.normal(mu, sigma)
                current_price *= np.exp(ret)
                all_paths[p, t] = current_price
                
        return all_paths

def run_simulation_demo(price, model, features):
    sim = MonteCarloSimulator(price, model, features)
    paths = sim.simulate()
    
    # Calculate bands
    p05 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    
    return p05, p50, p95, paths

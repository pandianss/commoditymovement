from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

class ElasticNetBaseline:
    """
    Standard ElasticNet baseline with scaling.
    """
    def __init__(self, alpha=0.001, l1_ratio=0.5):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000))
        ])
        
    def fit(self, X, y):
        # We handle NaN if any (though feature store should be clean)
        self.pipeline.fit(X, y)
        
    def predict(self, X):
        return self.pipeline.predict(X)

def get_baseline_model():
    return ElasticNetBaseline

from dataclasses import dataclass
from typing import Callable, Union
import numpy as np

@dataclass
class FeatureSpec:
    """
    Defines how a feature is computed and validated.
    """
    name: str
    source_col: str
    transformation: str # e.g., 'log', 'pct_change', 'lag', 'raw'
    params: dict = None
    
    def transform(self, series: Union[np.ndarray, 'pd.Series']):
        """
        Applies transformation.
        """
        if self.transformation == 'raw':
            return series
            
        if self.transformation == 'pct_change':
            return series.pct_change()
            
        if self.transformation == 'log':
            return np.log(series)
            
        if self.transformation == 'lag':
            lag = self.params.get('lag', 1)
            return series.shift(lag)
            
        raise ValueError(f"Unknown transformation: {self.transformation}")

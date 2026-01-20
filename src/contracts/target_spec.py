from dataclasses import dataclass

@dataclass
class TargetSpec:
    """
    Defines the prediction target to avoid ambiguity.
    """
    name: str # e.g., 'fwd_ret_5d'
    horizon_days: int
    type: str = 'regression' # 'regression' or 'classification'
    
    def __post_init__(self):
        if self.horizon_days <= 0:
            raise ValueError("Horizon must be positive.")

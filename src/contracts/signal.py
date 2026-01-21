from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any

@dataclass
class Signal:
    """
    Standardized predictive signal consumed by strategies.
    Supporting Requirement E.
    """
    timestamp_utc: datetime
    asset: str # Commodity Ticker
    signal_type: str # 'DIRECTIONAL', 'VOLATILITY', 'REGIME'
    direction: float # -1.0 (Short) to 1.0 (Long)
    probability: float # Confidence 0.0 to 1.0
    horizon: str # e.g. '5d', '1d'
    source: str # e.g. 'TCN_Model', 'News_Impact'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            'timestamp_utc': self.timestamp_utc,
            'asset': self.asset,
            'type': self.signal_type,
            'direction': self.direction,
            'prob': self.probability,
            'src': self.source
        }

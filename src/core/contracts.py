from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

class MarketDataContract(BaseModel):
    """Canonical schema for raw market data ingestion."""
    timestamp: datetime
    ticker: str
    open: float
    high: float
    low: float
    close: float
    adj_close: Optional[float] = None
    volume: int
    source: str = "yfinance"

class FeatureMetadata(BaseModel):
    """Metadata for versioned feature sets."""
    feature_set_id: str
    version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    columns: List[str]
    parameters: Dict[str, Union[int, float, str]]
    description: Optional[str] = None
    causal_offset: int = 1  # Standard shift count

class PredictionArtifact(BaseModel):
    """Schema for prediction outputs to ensure lineage."""
    timestamp: datetime
    model_id: str
    feature_set_version: str
    target: str
    median: float
    lower_bound: float
    upper_bound: float
    confidence_interval: float = 0.90
    metadata: Dict[str, Union[int, float, str]] = Field(default_factory=dict)

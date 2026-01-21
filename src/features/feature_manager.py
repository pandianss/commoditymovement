import os
import json
import pandas as pd
from typing import Optional
from core.contracts import FeatureMetadata
from datetime import datetime

class FeatureStoreManager:
    """
    Manages versioned feature artifacts.
    Enforces metadata binding and consistent naming conventions.
    """
    def __init__(self, base_path: str = "data/features"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def _get_paths(self, feature_set_id: str, version: str):
        base_name = f"{feature_set_id}_{version}"
        data_path = os.path.join(self.base_path, f"{base_name}.parquet")
        meta_path = os.path.join(self.base_path, f"{base_name}.json")
        return data_path, meta_path

    def save_feature_set(self, df: pd.DataFrame, metadata: FeatureMetadata):
        """
        Saves a dataframe as a versioned artifact with sidecar metadata.
        """
        data_path, meta_path = self._get_paths(metadata.feature_set_id, metadata.version)
        
        # Enforce timestamp index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Feature DataFrame must have a DatetimeIndex")
            
        # Save Data
        df.to_parquet(data_path)
        
        # Save Metadata
        with open(meta_path, 'w') as f:
            f.write(metadata.json(indent=4))
            
        print(f"Feature set saved: {data_path}")

    def load_feature_set(self, feature_set_id: str, version: str) -> hash:
        """
        Returns (DataFrame, FeatureMetadata)
        """
        data_path, meta_path = self._get_paths(feature_set_id, version)
        
        if not (os.path.exists(data_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(f"Feature set {feature_set_id} v{version} not found.")
            
        df = pd.read_parquet(data_path)
        with open(meta_path, 'r') as f:
            meta_json = json.load(f)
            metadata = FeatureMetadata(**meta_json)
            
        return df, metadata

    def list_versions(self, feature_set_id: str):
        # Implementation to scan dir could go here
        pass

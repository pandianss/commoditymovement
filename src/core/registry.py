import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class ModelMetadata(BaseModel):
    """Schema for model registry entries."""
    model_id: str = Field(..., description="Unique identifier (e.g., tcn_gold_v3)")
    version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    model_path: str
    feature_set_version: str
    feature_hash: str
    training_window: Dict[str, str]  # {"start": "2020-01-01", "end": "2023-12-31"}
    hyperparameters: Dict
    metrics: Dict[str, float]  # {"sharpe": 1.5, "mae": 0.02, "quantile_loss": 0.015}
    status: str = Field(default="candidate", description="candidate/champion/retired")
    promoted_at: Optional[datetime] = None
    notes: Optional[str] = None

class ModelRegistry:
    """
    Central registry for model governance.
    """
    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = registry_path
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        self.models = self._load()
    
    def _load(self) -> List[Dict]:
        """Load registry from disk."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return []
    
    def _save(self):
        """Persist registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.models, f, indent=4, default=str)
    
    def register(self, metadata: ModelMetadata) -> str:
        """
        Register a new model.
        Returns the model_id.
        """
        # Check for duplicate
        existing = self.get_model(metadata.model_id)
        if existing:
            raise ValueError(f"Model {metadata.model_id} already exists. Use update() or create new version.")
        
        self.models.append(metadata.dict())
        self._save()
        return metadata.model_id
    
    def get_model(self, model_id: str) -> Optional[Dict]:
        """Retrieve model metadata by ID."""
        for model in self.models:
            if model['model_id'] == model_id:
                return model
        return None
    
    def get_champion(self, model_family: str = "tcn_gold") -> Optional[Dict]:
        """
        Get the current champion model for a family.
        """
        champions = [m for m in self.models 
                    if m['model_id'].startswith(model_family) and m['status'] == 'champion']
        
        if not champions:
            return None
        
        # Return most recently promoted
        champions.sort(key=lambda x: x.get('promoted_at', ''), reverse=True)
        return champions[0]
    
    def promote_to_champion(self, model_id: str):
        """
        Promote a model to champion status.
        Automatically retires previous champion.
        """
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Extract family name (e.g., tcn_gold from tcn_gold_v3)
        family = '_'.join(model_id.split('_')[:-1])
        
        # Retire current champion
        current_champion = self.get_champion(family)
        if current_champion:
            for m in self.models:
                if m['model_id'] == current_champion['model_id']:
                    m['status'] = 'retired'
        
        # Promote new champion
        for m in self.models:
            if m['model_id'] == model_id:
                m['status'] = 'champion'
                m['promoted_at'] = datetime.utcnow().isoformat()
        
        self._save()
    
    def list_models(self, status: Optional[str] = None) -> List[Dict]:
        """List all models, optionally filtered by status."""
        if status:
            return [m for m in self.models if m['status'] == status]
        return self.models
    
    def update_metrics(self, model_id: str, new_metrics: Dict[str, float]):
        """Update metrics for a model (e.g., after live performance tracking)."""
        for m in self.models:
            if m['model_id'] == model_id:
                m['metrics'].update(new_metrics)
                self._save()
                return
        raise ValueError(f"Model {model_id} not found")

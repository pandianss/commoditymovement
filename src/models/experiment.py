import os
import json
from typing import Dict, Any
from datetime import datetime

class ExperimentLogger:
    """
    Tracks experiments during model training.
    """
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.experiment_dir = os.path.join(base_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.metadata = {
            "experiment_name": experiment_name,
            "started_at": datetime.utcnow().isoformat(),
            "hyperparameters": {},
            "metrics": {},
            "artifacts": []
        }
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        self.metadata["hyperparameters"].update(params)
        self._save()
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log training/validation metrics."""
        self.metadata["metrics"].update(metrics)
        self._save()
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "model"):
        """Record artifact location."""
        self.metadata["artifacts"].append({
            "path": artifact_path,
            "type": artifact_type,
            "logged_at": datetime.utcnow().isoformat()
        })
        self._save()
    
    def _save(self):
        """Persist experiment metadata."""
        metadata_path = os.path.join(self.experiment_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
    
    def finalize(self):
        """Mark experiment as complete."""
        self.metadata["completed_at"] = datetime.utcnow().isoformat()
        self._save()
        return self.metadata

import os
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd

class HealthMonitor:
    """
    Monitors system health across data, models, and infrastructure.
    """
    def __init__(self, state_dir: str = "state"):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.health_file = os.path.join(state_dir, "health_status.json")
        
    def check_data_freshness(self, last_update_time: datetime, max_age_hours: int = 2) -> Dict:
        """
        Checks if data is stale.
        """
        age = datetime.utcnow() - last_update_time
        is_fresh = age < timedelta(hours=max_age_hours)
        
        return {
            "component": "data_freshness",
            "status": "healthy" if is_fresh else "stale",
            "last_update": last_update_time.isoformat(),
            "age_hours": age.total_seconds() / 3600,
            "threshold_hours": max_age_hours
        }
    
    def check_model_health(self, recent_predictions: pd.DataFrame) -> Dict:
        """
        Basic model health check - ensures predictions are within reasonable bounds.
        """
        if recent_predictions.empty:
            return {
                "component": "model_health",
                "status": "unknown",
                "reason": "no_recent_predictions"
            }
        
        # Check for NaN predictions
        has_nans = recent_predictions.isnull().any().any()
        
        # Check prediction variance (too low = model stuck, too high = unstable)
        if 0.5 in recent_predictions.columns:
            median_preds = recent_predictions[0.5]
            variance = median_preds.var()
            
            status = "healthy"
            if variance < 1e-6:
                status = "degraded"
                reason = "predictions_constant"
            elif variance > 0.1:
                status = "unstable"
                reason = "high_variance"
            elif has_nans:
                status = "error"
                reason = "nan_predictions"
            else:
                reason = "normal"
                
            return {
                "component": "model_health",
                "status": status,
                "variance": float(variance),
                "reason": reason
            }
        
        return {"component": "model_health", "status": "unknown"}
    
    def record_heartbeat(self):
        """
        Records a heartbeat timestamp.
        """
        heartbeat = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "alive"
        }
        
        with open(self.health_file, 'w') as f:
            json.dump(heartbeat, f, indent=4)
    
    def get_system_status(self) -> Dict:
        """
        Returns overall system health status.
        """
        if not os.path.exists(self.health_file):
            return {"status": "unknown", "reason": "no_heartbeat_file"}
        
        with open(self.health_file, 'r') as f:
            return json.load(f)

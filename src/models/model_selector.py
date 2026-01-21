from typing import Dict, Optional
from core.registry import ModelRegistry

class ModelSelector:
    """
    Manages champion/challenger logic and model promotion.
    """
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
    
    def compare_models(self, candidate_id: str, champion_id: str, 
                      primary_metric: str = "sharpe") -> Dict:
        """
        Compare candidate vs champion on primary metric.
        Returns comparison results and recommendation.
        """
        candidate = self.registry.get_model(candidate_id)
        champion = self.registry.get_model(champion_id)
        
        if not candidate or not champion:
            raise ValueError("Both models must exist in registry")
        
        candidate_score = candidate['metrics'].get(primary_metric, 0)
        champion_score = champion['metrics'].get(primary_metric, 0)
        
        improvement = (candidate_score - champion_score) / abs(champion_score) if champion_score != 0 else 0
        
        # Decision logic: require >5% improvement to promote
        should_promote = improvement > 0.05
        
        return {
            "candidate_id": candidate_id,
            "champion_id": champion_id,
            "candidate_score": candidate_score,
            "champion_score": champion_score,
            "improvement_pct": improvement * 100,
            "recommendation": "promote" if should_promote else "keep_champion",
            "reason": f"{improvement*100:.2f}% improvement" if should_promote else "insufficient improvement"
        }
    
    def auto_promote_if_better(self, candidate_id: str, 
                               model_family: str = "tcn_gold",
                               primary_metric: str = "sharpe") -> bool:
        """
        Automatically promote candidate if it outperforms current champion.
        Returns True if promoted.
        """
        champion = self.registry.get_champion(model_family)
        
        if not champion:
            # No champion exists, promote candidate
            self.registry.promote_to_champion(candidate_id)
            return True
        
        comparison = self.compare_models(candidate_id, champion['model_id'], primary_metric)
        
        if comparison['recommendation'] == 'promote':
            self.registry.promote_to_champion(candidate_id)
            return True
        
        return False
    
    def rollback_to_previous(self, model_family: str = "tcn_gold"):
        """
        Rollback to the most recent retired model.
        Useful if current champion underperforms in production.
        """
        retired_models = [m for m in self.registry.list_models(status='retired')
                         if m['model_id'].startswith(model_family)]
        
        if not retired_models:
            raise ValueError(f"No retired models found for {model_family}")
        
        # Get most recently retired
        retired_models.sort(key=lambda x: x.get('promoted_at', ''), reverse=True)
        rollback_model = retired_models[0]
        
        self.registry.promote_to_champion(rollback_model['model_id'])
        return rollback_model['model_id']

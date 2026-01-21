import unittest
import os
import json
import tempfile
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.registry import ModelRegistry, ModelMetadata
from models.model_selector import ModelSelector
from datetime import datetime

class TestModelRegistry(unittest.TestCase):
    
    def setUp(self):
        # Create temporary registry file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        self.temp_path = self.temp_file.name
        self.temp_file.close()
        self.registry = ModelRegistry(self.temp_path)
    
    def tearDown(self):
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)
    
    def test_register_model(self):
        """Test model registration."""
        metadata = ModelMetadata(
            model_id="test_model_v1",
            version="1.0",
            model_path="/path/to/model.pth",
            feature_set_version="v1",
            feature_hash="abc123",
            training_window={"start": "2020-01-01", "end": "2023-12-31"},
            hyperparameters={"lr": 0.001},
            metrics={"val_loss": 0.05}
        )
        
        model_id = self.registry.register(metadata)
        self.assertEqual(model_id, "test_model_v1")
        
        # Verify retrieval
        retrieved = self.registry.get_model("test_model_v1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved['metrics']['val_loss'], 0.05)
    
    def test_champion_promotion(self):
        """Test champion promotion logic."""
        # Register two models
        model1 = ModelMetadata(
            model_id="tcn_gold_v1",
            version="1.0",
            model_path="/path/to/v1.pth",
            feature_set_version="v1",
            feature_hash="abc123",
            training_window={"start": "2020-01-01", "end": "2023-12-31"},
            hyperparameters={},
            metrics={"val_loss": 0.10}
        )
        
        model2 = ModelMetadata(
            model_id="tcn_gold_v2",
            version="1.0",
            model_path="/path/to/v2.pth",
            feature_set_version="v1",
            feature_hash="abc123",
            training_window={"start": "2020-01-01", "end": "2023-12-31"},
            hyperparameters={},
            metrics={"val_loss": 0.08}
        )
        
        self.registry.register(model1)
        self.registry.register(model2)
        
        # Promote v1 to champion
        self.registry.promote_to_champion("tcn_gold_v1")
        champion = self.registry.get_champion("tcn_gold")
        self.assertEqual(champion['model_id'], "tcn_gold_v1")
        
        # Promote v2 - should retire v1
        self.registry.promote_to_champion("tcn_gold_v2")
        new_champion = self.registry.get_champion("tcn_gold")
        self.assertEqual(new_champion['model_id'], "tcn_gold_v2")
        
        # Verify v1 is retired
        v1 = self.registry.get_model("tcn_gold_v1")
        self.assertEqual(v1['status'], 'retired')
    
    def test_model_selector_comparison(self):
        """Test model comparison logic."""
        # Register models
        champion = ModelMetadata(
            model_id="tcn_gold_champion",
            version="1.0",
            model_path="/path/to/champion.pth",
            feature_set_version="v1",
            feature_hash="abc123",
            training_window={"start": "2020-01-01", "end": "2023-12-31"},
            hyperparameters={},
            metrics={"sharpe": 1.0}
        )
        
        candidate = ModelMetadata(
            model_id="tcn_gold_candidate",
            version="1.0",
            model_path="/path/to/candidate.pth",
            feature_set_version="v1",
            feature_hash="abc123",
            training_window={"start": "2020-01-01", "end": "2023-12-31"},
            hyperparameters={},
            metrics={"sharpe": 1.2}  # 20% improvement
        )
        
        self.registry.register(champion)
        self.registry.register(candidate)
        
        selector = ModelSelector(self.registry)
        comparison = selector.compare_models("tcn_gold_candidate", "tcn_gold_champion", "sharpe")
        
        self.assertEqual(comparison['recommendation'], 'promote')
        self.assertGreater(comparison['improvement_pct'], 5)

if __name__ == '__main__':
    unittest.main()

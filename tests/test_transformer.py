import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.transformer_core import TransformerQuantileModel

class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.seq_len = 20
        self.input_dim = 10
        self.d_model = 16
        self.quantiles = [0.05, 0.5, 0.95]
        
    def test_transformer_forward(self):
        """Test Transformer Quantile Model forward pass"""
        model = TransformerQuantileModel(
            input_dim=self.input_dim, 
            d_model=self.d_model, 
            nhead=2, 
            quantiles=self.quantiles
        )
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        
        preds = model(x)
        
        # Check output shapes
        for q in self.quantiles:
            self.assertIn(q, preds)
            self.assertEqual(preds[q].shape, (self.batch_size, 1))

if __name__ == '__main__':
    unittest.main()

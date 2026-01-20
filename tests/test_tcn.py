import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.tcn_engine import TCNQuantileModel, TemporalConvolutionalNetwork

class TestTCN(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.seq_len = 20
        self.input_size = 10
        self.num_channels = [16, 32, 16]
        self.quantiles = [0.05, 0.5, 0.95]

    def test_tcn_output_shape(self):
        """Test the raw TCN network output shape"""
        model = TemporalConvolutionalNetwork(self.input_size, self.num_channels)
        x = torch.randn(self.batch_size, self.seq_len, self.input_size)
        
        # TCN forward expects (N, L, C) -> returns (N, L, out_channels[-1])
        output = model(x)
        expected_shape = (self.batch_size, self.seq_len, self.num_channels[-1])
        self.assertEqual(output.shape, expected_shape)

    def test_quantile_model_forward(self):
        """Test the full Quantile Model forward pass"""
        model = TCNQuantileModel(self.input_size, self.num_channels, quantiles=self.quantiles)
        x = torch.randn(self.batch_size, self.seq_len, self.input_size)
        
        preds = model(x)
        
        # Check if we get predictions for all quantiles
        for q in self.quantiles:
            self.assertIn(q, preds)
            # Each prediction should be (N, 1) or (N) depending on logic, let's check code
            # Code does: self.heads[...](last_step) where last_step is (N, C) -> Linear -> (N, 1)
            self.assertEqual(preds[q].shape, (self.batch_size, 1))

if __name__ == '__main__':
    unittest.main()

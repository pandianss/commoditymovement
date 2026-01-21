import unittest
import pandas as pd
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.time_index import causal_slice, apply_causal_mask
from core.contracts import FeatureMetadata
from models.validation import WalkForwardSplitter

class TestCausalCore(unittest.TestCase):
    def test_causal_slice(self):
        dates = pd.date_range(start="2023-01-01", periods=5, freq='D')
        df = pd.DataFrame({'value': range(5)}, index=dates)
        
        # Slice at index 2 (2023-01-03)
        cutoff = datetime(2023, 1, 3)
        sliced = causal_slice(df, cutoff)
        
        self.assertEqual(len(sliced), 3)
        self.assertEqual(sliced.index.max(), cutoff)
        self.assertNotIn(4, sliced['value'].values)

    def test_apply_causal_mask(self):
        df = pd.DataFrame({'value': [1, 2, 3]})
        shifted = apply_causal_mask(df, shift_count=1)
        
        self.assertTrue(pd.isna(shifted.iloc[0, 0]))
        self.assertEqual(shifted.iloc[1, 0], 1.0)

    def test_feature_metadata_validation(self):
        # Valid metadata
        valid = FeatureMetadata(
            feature_set_id="test_set",
            version="v1",
            columns=["a", "b"],
            parameters={"window": 10},
            causal_offset=1
        )
        self.assertEqual(valid.version, "v1")

        # Invalid metadata (missing required field)
        with self.assertRaises(Exception):
            FeatureMetadata(feature_set_id="test_set")

    def test_walk_forward_splitter(self):
        X = pd.DataFrame({'val': range(20)})
        splitter = WalkForwardSplitter(n_splits=3, test_size=2, embargo=1)
        
        splits = list(splitter.split(X))
        self.assertEqual(len(splits), 3)
        
        for train_idx, test_idx in splits:
            self.assertLess(max(train_idx), min(test_idx))
            # Verify no overlap
            self.assertEqual(len(set(train_idx).intersection(set(test_idx))), 0)

if __name__ == '__main__':
    unittest.main()

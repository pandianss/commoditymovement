import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from news_engine.nlp_processor import NLPProcessor
from research.correlation_discovery import CorrelationDiscovery

class TestNewsIntelligence(unittest.TestCase):
    def test_sentiment_aggregation(self):
        processor = NLPProcessor(backend="vader")
        dates = pd.date_range(start="2023-01-01", periods=10, freq='10min')
        df = pd.DataFrame({
            'headline': ['Gold is GREAT!'] * 10
        }, index=dates)
        
        processed = processor.process_headlines(df)
        agg = processor.aggregate_sentiment(processed, freq='1H')
        
        self.assertEqual(len(agg), 2) # 100 mins -> 2 hours
        self.assertIn('sent_mean', agg.columns)
        self.assertGreater(agg['sent_mean'].iloc[0], 0)

    def test_lead_lag_discovery(self):
        """
        Verify that the engine discovers a known lead signal.
        """
        discovery = CorrelationDiscovery(max_lag_periods=5)
        
        # Create a signal that leads a target by 2 periods
        n = 100
        signal = pd.Series(np.random.normal(0, 1, n))
        # Target(T) = Signal(T-2) + noise
        target = signal.shift(2).fillna(0) + np.random.normal(0, 0.1, n)
        
        # Add index
        idx = pd.date_range(start="2023-01-01", periods=n, freq='H')
        signal.index = idx
        target.index = idx
        
        lag, corr = discovery.find_best_lead(signal, target)
        
        # Discovery should find lag=2 (meaning signal from T-2 leads target at T)
        self.assertEqual(lag, 2)
        self.assertGreater(corr, 0.8)

if __name__ == '__main__':
    unittest.main()

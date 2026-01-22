import pytest
import pandas as pd
import numpy as np
import os
import sys

# Mock src in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.intelligence.event_alignment import EventAligner
from src.intelligence.impact_analyzer import ImpactAnalyzer

def test_know_positive_impact():
    """
    Tier 3: Intelligence Verification
    Verify that a 'perfectly positive' synthetic news event generates positive impact stats.
    """
    # 1. Create Bullish Market
    dates = pd.date_range("2020-01-01", periods=20, freq='D')
    prices = np.linspace(100, 110, 20) # Steady rise
    market_df = pd.DataFrame({'close': prices, 'commodity': 'GOLD'}, index=dates)
    
    # 2. Create Positive News in middle
    news_date = dates[10]
    news_df = pd.DataFrame([{
        'timestamp_utc': news_date,
        'headline': 'Gold finds huge deposit',
        'sentiment': 'positive',
        'commodity': 'GOLD',
        'source': 'Test'
    }])
    
    # 3. Align
    aligner = EventAligner(market_df, news_df)
    aligned = aligner.align_to_windows(lookback_days=1, lookforward_days=5)
    
    # 4. Analyze
    analyzer = ImpactAnalyzer(aligned)
    stats = analyzer.compute_conditional_returns(group_by=['sentiment'], horizon='fwd_ret_5d')
    
    # 5. Assert Positive Return
    mean_ret = stats.loc['positive', 'mean']
    assert mean_ret > 0, "Positive news in a bull market should show positive impact"

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelligence.event_alignment import EventAligner
from intelligence.impact_analyzer import ImpactAnalyzer

def create_dummy_market_data():
    # Create 30 days of data around the event dates in sample
    # Sample dates: 2020-08-01, 2022-03-15
    dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq='B')
    df = pd.DataFrame(index=dates)
    df['commodity'] = 'GOLD'
    # Random walk
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, len(dates))
    df['close'] = 1000 * (1 + returns).cumprod()
    return df

def main():
    print("--- Market Intelligence Core Demo ---")
    
    # 1. Load Sample News
    news_path = "data/raw/gold-dataset-sample.csv"
    if not os.path.exists(news_path):
        print(f"Sample news not found at {news_path}")
        return
        
    news_df = pd.read_csv(news_path)
    news_df['timestamp_utc'] = pd.to_datetime(news_df['date']) # Sample has 'date'
    news_df['commodity'] = 'GOLD' # Ensure commodity col exists
    print(f"Loaded {len(news_df)} news items.")
    
    # 2. Load Market Data (Dummy)
    market_df = create_dummy_market_data()
    print(f"Generated market data: {len(market_df)} days.")
    
    # 3. Running Event Aligner
    print("\nRunning Event Aligner...")
    aligner = EventAligner(market_df, news_df)
    aligned_events = aligner.align_to_windows(lookback_days=1, lookforward_days=5)
    
    if aligned_events.empty:
        print("No events aligned.")
        return
        
    print(f"Aligned {len(aligned_events)} events.")
    print(aligned_events[['headline', 'fwd_ret_1d', 'fwd_ret_5d']].head())
    
    # 4. Running Impact Analyzer
    print("\nRunning Impact Analyzer...")
    analyzer = ImpactAnalyzer(aligned_events)
    
    # Conditional Returns by Sentiment
    print("\nConditional Returns by Sentiment:")
    stats = analyzer.compute_conditional_returns(group_by=['sentiment'], horizon='fwd_ret_5d')
    print(stats)
    
    # Shock Probability
    prob = analyzer.compute_shock_probability(threshold=0.01, horizon='fwd_ret_5d')
    print(f"\nShock Probability (>1% move in 5 days): {prob:.2%}")
    
    # 5. Running Causality Engine
    print("\nRunning Causality Engine...")
    from intelligence.causality import CausalityEngine
    
    # Map sentiment to score for numerical analysis
    sent_map = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
    if 'sentiment_score' not in news_df.columns:
        news_df['sentiment_score'] = news_df['sentiment'].map(sent_map)
        
    causality = CausalityEngine(market_df, news_df)
    # Test Granger (Gold) - might fail if not enough data points in intersection, but logic check
    p_values = causality.test_granger_causality('GOLD', max_lag=2)
    print("Granger Causality (p-values by lag):")
    if p_values:
        for lag, p in p_values.items():
            print(f"  Lag {lag}: p={p:.4f}")
    else:
        print("  Not enough congruent daily data for Granger test (Requires overlapping dates).")

if __name__ == "__main__":
    main()

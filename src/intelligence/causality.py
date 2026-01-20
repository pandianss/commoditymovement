import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import logging

logger = logging.getLogger(__name__)

class CausalityEngine:
    """
    Identifies causal links between news sentiment and market moves.
    Supporting Requirement B.
    """
    
    def __init__(self, market_data: pd.DataFrame, news_data: pd.DataFrame):
        self.market_data = market_data
        self.news_data = news_data
        
    def test_granger_causality(self, commodity, max_lag=5):
        """
        Tests if news sentiment Granger-causes market returns.
        """
        try:
            # Prepare dataset
            # 1. Resample news to daily mean sentiment
            comm_news = self.news_data[self.news_data['commodity'] == commodity]
            if comm_news.empty:
                return {}
                
            daily_sentiment = comm_news.set_index('timestamp_utc').resample('B')['sentiment_score'].mean()
            
            # 2. Get market returns
            comm_market = self.market_data[self.market_data['commodity'] == commodity]
            daily_returns = comm_market['close'].pct_change().dropna()
            
            # 3. Align
            df = pd.concat([daily_sentiment, daily_returns], axis=1).dropna()
            df.columns = ['sentiment', 'returns']
            
            if len(df) < max_lag + 5:
                # Not enough data
                return {}
                
            # Run Test (returns ~ sentiment)
            # checks if sentiment causes returns
            res = grangercausalitytests(df[['returns', 'sentiment']], maxlag=max_lag, verbose=False)
            
            # Extract p-values for each lag
            p_values = {lag: res[lag][0]['ssr_ftest'][1] for lag in res}
            return p_values
            
        except Exception as e:
            logger.error(f"Granger test failed: {e}")
            return {}

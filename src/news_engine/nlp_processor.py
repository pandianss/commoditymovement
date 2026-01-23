import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class NLPProcessor:
    """
    Unified interface for news processing and sentiment extraction.
    Supports pluggable backends and causal batch processing.
    """
    def __init__(self, backend: str = "finbert"):
        self.backend = backend
        self.analyzer = None
        self.model_loaded = False
        
        if backend == "finbert" and TRANSFORMERS_AVAILABLE:
            try:
                # ProsusAI/finbert is the standard for financial sentiment
                self.pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
                self.model_loaded = True
            except Exception as e:
                print(f"Failed to load FinBERT: {e}. Falling back to VADER.")
                self.backend = "vader"
        
        if not self.model_loaded or self.backend == "vader":
            self.analyzer = SentimentIntensityAnalyzer()
            self.backend = "vader"

    def get_sentiment(self, text: str) -> float:
        """
        Extracts a scalar sentiment score (-1 to 1).
        """
        if self.backend == "finbert" and self.model_loaded:
            try:
                result = self.pipe(text)[0]
                label = result['label'] # 'positive', 'negative', 'neutral'
                score = result['score']
                
                if label == 'positive':
                    return score
                elif label == 'negative':
                    return -score
                else:
                    return 0.0
            except Exception as e:
                # Silent fallback to neutral or logs if needed
                return 0.0
                
        if self.backend == "vader" and self.analyzer:
            score = self.analyzer.polarity_scores(text)
            return score['compound']
        return 0.0

    def process_headlines(self, df: pd.DataFrame, headline_col: str = 'headline') -> pd.DataFrame:
        """
        Applies sentiment extraction to a dataframe of headlines.
        """
        df = df.copy()
        df['sentiment_score'] = df[headline_col].apply(self.get_sentiment)
        return df

    def aggregate_sentiment(self, df: pd.DataFrame, freq: str = '1H') -> pd.DataFrame:
        """
        Aggregates news-level sentiment into a time-series.
        Expects index to be pd.DatetimeIndex.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Dataframe index must be DatetimeIndex for aggregation.")
            
        # Count articles and mean sentiment
        agg = df.resample(freq).agg({
            'sentiment_score': ['mean', 'count', 'std']
        })
        
        # Flatten columns
        agg.columns = [f"sent_{c[1]}" for c in agg.columns]
        
        # Fill gaps (0 sentiment if no news)
        agg['sent_mean'] = agg['sent_mean'].fillna(0)
        agg['sent_count'] = agg['sent_count'].fillna(0)
        agg['sent_std'] = agg['sent_std'].fillna(0)
        
        return agg

class EntityExtractor:
    """
    Experimental: Simple rule-based entity extraction for commodity keywords.
    """
    COMMODITY_KEYWORDS = {
        'gold': ['gold', 'xau', 'bullion'],
        'crude': ['crude', 'oil', 'wti', 'brent'],
        'copper': ['copper', 'hg=f'],
        'gas': ['natural gas', 'ng=f'],
        'silver': ['silver', 'xag']
    }

    def extract_mentions(self, text: str) -> List[str]:
        text_lower = text.lower()
        mentions = []
        for commodity, keywords in self.COMMODITY_KEYWORDS.items():
            if any(k in text_lower for k in keywords):
                mentions.append(commodity)
        return mentions

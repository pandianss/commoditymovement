import pandas as pd
from typing import List, Dict
from news_engine.nlp_processor import NLPProcessor, EntityExtractor

class NewsSignalDiscovery:
    """
    Synthesizes signals from NLP processing and topic modeling.
    """
    def __init__(self, nlp_processor: NLPProcessor):
        self.processor = nlp_processor
        self.extractor = EntityExtractor()

    def discover_high_intensity_signals(self, news_df: pd.DataFrame, sentiment_threshold: float = 0.5) -> pd.DataFrame:
        """
        Filters news for high-intensity signals that mention specific entities.
        """
        # Ensure sentiment is present
        if 'sentiment_score' not in news_df.columns:
            news_df = self.processor.process_headlines(news_df)
            
        # Extract entities
        news_df['mentions'] = news_df['headline'].apply(self.extractor.extract_mentions)
        
        # Filter for shocks (high intensity)
        shocks = news_df[news_df['sentiment_score'].abs() > sentiment_threshold].copy()
        
        # Explode mentions to get one row per commodity mention
        shocks = shocks.explode('mentions')
        shocks = shocks.dropna(subset=['mentions'])
        
        return shocks

    def generate_topic_sentiment_map(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates average sentiment per topic.
        """
        if 'topic_id' not in news_df.columns:
            return pd.DataFrame()
            
        return news_df.groupby('topic_id').agg({
            'sentiment_score': ['mean', 'count', 'std'],
            'topic_confidence': 'mean'
        })

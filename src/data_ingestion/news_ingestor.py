import os
import pandas as pd
from abc import ABC, abstractmethod

class NewsProvider(ABC):
    @abstractmethod
    def fetch_news(self, start_date, end_date, tickers=None):
        pass

class MockNewsProvider(NewsProvider):
    """
    Generates synthetic but relevant news for testing the pipeline 
    on historical inflection points.
    """
    def __init__(self):
        self.mock_events = {
            "2020-04-20": [
                {"headline": "Oil prices crash into negative territory as demand vanishes", "source": "Reuters", "commodity": "CRUDE_OIL"},
                {"headline": "WTI crude futures drop below zero in historic first", "source": "Bloomberg", "commodity": "CRUDE_OIL"}
            ],
            "2022-02-24": [
                {"headline": "Russia invades Ukraine, triggering global commodity spike", "source": "AP", "commodity": "GOLD"},
                {"headline": "Safe-haven gold surges as geopolitical tensions erupt", "source": "CNBC", "commodity": "GOLD"}
            ],
            "2011-09-05": [
                {"headline": "Gold hits all-time high of $1,900 as debt crisis looms", "source": "FT", "commodity": "GOLD"}
            ]
        }

    def fetch_news(self, start_date, end_date, tickers=None):
        # Filter mock events by date range
        news = []
        for date_str, items in self.mock_events.items():
            if start_date <= date_str <= end_date:
                for item in items:
                    item['timestamp_utc'] = date_str + " 12:00:00"
                    news.append(item)
        return pd.DataFrame(news)

class HistoricalNewsIngestor(NewsProvider):
    """
    Ingests historical news from a CSV file (e.g., Kaggle Gold Dataset).
    Expected columns: headline/text, sentiment (optional), date
    """
    def __init__(self, csv_path):
        self.csv_path = csv_path
        
    def fetch_news(self, start_date, end_date, tickers=None):
        if not os.path.exists(self.csv_path):
            print(f"Historical file not found at {self.csv_path}")
            return pd.DataFrame()
            
        df = pd.read_csv(self.csv_path)
        
        # Normalize columns
        # Map unique columns from potential datasets
        col_map = {
            'text': 'headline',
            'news': 'headline', 
            'date': 'timestamp_utc',
            'label': 'sentiment'
        }
        df = df.rename(columns=col_map)
        
        # Ensure required columns
        if 'headline' not in df.columns:
            print("Error: CSV must contain 'headline', 'text', or 'news' column.")
            return pd.DataFrame()
            
        if 'timestamp_utc' not in df.columns:
            # If no date, generate dummy or fail. For this specific dataset, we might need to parse.
            # The sample has 'date'.
            pass
            
        # Filter by date
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        mask = (df['timestamp_utc'] >= pd.to_datetime(start_date)) & (df['timestamp_utc'] <= pd.to_datetime(end_date))
        df = df[mask]
        
        # Add metadata
        if 'source' not in df.columns:
            df['source'] = 'Historical_Dataset'
        if 'commodity' not in df.columns:
            df['commodity'] = 'GOLD' # Default for this dataset
            
        return df

class NewsIngestionManager:
    def __init__(self, provider: NewsProvider):
        self.provider = provider
        
    def ingest(self, start_date, end_date, output_path):
        print(f"Ingesting news from {start_date} to {end_date}...")
        news_df = self.provider.fetch_news(start_date, end_date)
        if not news_df.empty:
            news_df.to_csv(output_path, index=False)
            print(f"Saved {len(news_df)} news items to {output_path}")
        else:
            print("No news found for the given range.")
        return news_df

if __name__ == "__main__":
    # Example usage with Mock Provider
    manager = NewsIngestionManager(MockNewsProvider())
    manager.ingest("2010-01-01", "2024-12-31", "data/raw/news_raw.csv")

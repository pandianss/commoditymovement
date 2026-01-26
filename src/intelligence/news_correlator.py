import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from data_ingestion.news_ingestor import NewsIngestionManager, MockNewsProvider

logger = setup_logger("news_correlator", log_file="news_correlator.log")

class NewsCorrelator:
    """
    Correlates price anomalies with historical news events.
    Uses a Knowledge Base for major historical events and API for recent ones.
    """
    
    # Knowledge Base of Major Market Events (Fallback for old data)
    MAJOR_EVENTS = {
        "2008-01-21": "Global Market Melt-down (Black Monday)",
        "2008-09-15": "Lehman Brothers Bankruptcy",
        "2008-10-24": "Global Financial Crisis Panic Selling",
        "2016-11-09": "US Election Volatility (Trump & Brexit Era)",
        "2020-03-09": "COVID-19 Crash (Black Monday I)",
        "2020-03-12": "COVID-19 Crash (Black Thursday)",
        "2020-03-23": "COVID-19 Market Bottom / Fed Stimulus",
        "2022-02-24": "Russia-Ukraine War Invasion",
        "2022-06-13": "US CPI Inflation Shock (Pre-Fed Hike)",
        "2024-06-04": "India Election Results Volatility"
    }

    def __init__(self, news_provider=None):
        self.news_provider = news_provider

    def correlate_spikes(self, spikes_df):
        """
        Annotates spikes with potential news causes.
        """
        if spikes_df.empty:
            return spikes_df

        spikes_df['Event_Context'] = None
        
        logger.info("Correlating spikes with Knowledge Base and News...")
        
        for index, row in spikes_df.iterrows():
            date_str = str(row['DATE'])
            
            # 1. Check Knowledge Base (Exact or +/- 1 day)
            context = self._check_knowledge_base(date_str)
            
            # 2. If no KB match and provider available, query API (Simulated here for now)
            # In production, you would call self.news_provider.fetch_news(date)
            
            if context:
                spikes_df.at[index, 'Event_Context'] = context
                
        # Fill remaining with generic description based on move
        mask = spikes_df['Event_Context'].isnull()
        spikes_df.loc[mask, 'Event_Context'] = spikes_df.loc[mask].apply(
            lambda x: f"Significant {x['Type']} ({x['Returns']:.1%})", axis=1
        )
        
        return spikes_df

    def _check_knowledge_base(self, date_str):
        """Checks if date is near a known major event."""
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        
        # Check exact and surrounding dates
        for offset in [-1, 0, 1]:
            check_date = (target_date + timedelta(days=offset)).strftime("%Y-%m-%d")
            if check_date in self.MAJOR_EVENTS:
                return self.MAJOR_EVENTS[check_date]
        return None

if __name__ == "__main__":
    # Test
    try:
        spikes_path = "data/processed/detected_spikes.csv"
        if os.path.exists(spikes_path):
            df = pd.read_csv(spikes_path)
            correlator = NewsCorrelator()
            correlated_df = correlator.correlate_spikes(df)
            print(correlated_df.head(10))
            correlated_df.to_csv("data/processed/spikes_with_context.csv", index=False)
            print("Saved context to data/processed/spikes_with_context.csv")
    except Exception as e:
        print(f"Error testing correlator: {e}")

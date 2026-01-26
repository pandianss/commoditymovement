import requests
import pandas as pd
import os
import sys
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_ingestion.news_ingestor import NewsProvider
from utils.logger import setup_logger

logger = setup_logger("alpha_vantage", log_file="news_ingestion.log")

class AlphaVantageNewsProvider(NewsProvider):
    """
    Fetches real-time news sentiment from Alpha Vantage.
    Optimized for large universes with sector-based batching.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # Free tier: 5 calls/min = 12s between calls

    def fetch_news(self, start_date=None, end_date=None, tickers=None, time_from=None):
        """
        Alpha Vantage News API implementation.
        time_from: String in YYYYMMDDTHHMM format to fetch only latest news.
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key,
            "sort": "LATEST",
            "limit": 200
        }
        
        if time_from:
            params["time_from"] = time_from
            logger.info(f"Polling Alpha Vantage News since {time_from}...")
        
        if tickers:
            params["tickers"] = ",".join(tickers)
        else:
            params["topics"] = "technology,finance,energy"
            
        logger.info(f"Requesting Alpha Vantage News for {params.get('tickers', params.get('topics'))}...")
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error fetching news from Alpha Vantage: {e}")
            return pd.DataFrame()
            
        data = response.json()
        
        # Check for Alpha Vantage specific error/limit messages
        if "Note" in data:
            logger.warning(f"Alpha Vantage API Note: {data['Note']}")
            return pd.DataFrame()
        
        if "ErrorMessage" in data:
            logger.error(f"Alpha Vantage API Error: {data['ErrorMessage']}")
            return pd.DataFrame()

        if "feed" not in data:
            logger.debug(f"No news feed found in response. Possible zero results or rate limit.")
            return pd.DataFrame()
            
        news_items = []
        for item in data["feed"]:
            news_items.append({
                "headline": item.get("title"),
                "source": item.get("source"),
                "timestamp_utc": item.get("time_published"),
                "relevance_score": item.get("overall_sentiment_score"),
                "summary": item.get("summary"),
                "url": item.get("url")
            })
            
        logger.info(f"Successfully fetched {len(news_items)} news items.")
        return pd.DataFrame(news_items)

    def fetch_news_by_sectors(self, sectors, time_from=None):
        """
        Fetch news for multiple sectors with rate limiting.
        sectors: List of sector names (e.g., ['technology', 'finance', 'energy'])
        """
        all_news = []
        
        for i, sector in enumerate(sectors):
            logger.info(f"Fetching news for sector: {sector} ({i+1}/{len(sectors)})")
            
            params = {
                "function": "NEWS_SENTIMENT",
                "apikey": self.api_key,
                "topics": sector,
                "sort": "LATEST",
                "limit": 200
            }
            
            if time_from:
                params["time_from"] = time_from
            
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if "Note" in data or "ErrorMessage" in data:
                    logger.warning(f"API limit or error for {sector}: {data.get('Note', data.get('ErrorMessage'))}")
                    continue
                
                if "feed" in data:
                    for item in data["feed"]:
                        all_news.append({
                            "headline": item.get("title"),
                            "source": item.get("source"),
                            "timestamp_utc": item.get("time_published"),
                            "relevance_score": item.get("overall_sentiment_score"),
                            "summary": item.get("summary"),
                            "url": item.get("url"),
                            "sector": sector
                        })
                    logger.info(f"Fetched {len(data['feed'])} items for {sector}")
                
                # Rate limiting
                if i < len(sectors) - 1:
                    logger.debug(f"Rate limit delay: {self.rate_limit_delay}s")
                    time.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                logger.error(f"Error fetching news for {sector}: {e}")
                continue
        
        logger.info(f"Total news items fetched: {len(all_news)}")
        return pd.DataFrame(all_news)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if api_key:
        provider = AlphaVantageNewsProvider(api_key)
        # Test sector-based fetching
        sectors = ['technology', 'finance', 'energy']
        df = provider.fetch_news_by_sectors(sectors)
        print(f"Fetched {len(df)} news items")
        print(df.head())
    else:
        logger.error("Set ALPHA_VANTAGE_API_KEY in .env to test.")

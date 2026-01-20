import requests
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_ingestion.news_ingestor import NewsProvider
from utils.logger import logger

class AlphaVantageNewsProvider(NewsProvider):
    """
    Fetches real-time news sentiment from Alpha Vantage.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

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
            params["topics"] = "commodities"
            
        logger.info(f"Requesting Alpha Vantage News for {params.get('tickers', 'commodities')}...")
        
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

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if api_key:
        provider = AlphaVantageNewsProvider(api_key)
        df = provider.fetch_news()
        print(df.head())
    else:
        logger.error("Set ALPHA_VANTAGE_API_KEY in .env to test.")

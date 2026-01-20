import time
import os
import sys
import datetime
import requests
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, COMMODITIES
from data_ingestion.market_data import fetch_intraday_data
from data_ingestion.alpha_vantage_provider import AlphaVantageNewsProvider
from news_engine.realtime_shocks import detect_intraday_shocks
from utils.logger import setup_logger

logger = setup_logger("live_intelligence", log_file="live_intel.log")

def run_intra_day_loop(poll_interval_mins=30):
    load_dotenv()
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        logger.warning("No ALPHA_VANTAGE_API_KEY found. Intra-day polling will be limited.")
        
    provider = AlphaVantageNewsProvider(api_key) if api_key else None
    
    logger.info("Starting Intra-day Intelligence Loop...")
    
    while True:
        try:
            now = datetime.datetime.now()
            logger.info(f"--- Cycle Start: {now.strftime('%H:%M:%S')} ---")
            
            # 1. Update Intra-day Prices (1h)
            logger.debug("Fetching 1h intraday data...")
            df_1h = fetch_intraday_data(COMMODITIES, interval="1h", period="5d")
            
            # 2. Detect Shocks
            if not df_1h.empty:
                shocks = detect_intraday_shocks(df_1h)
                if not shocks.empty:
                    latest_shock = shocks.iloc[-1]
                    logger.info(f"!!! SHOCK DETECTED: {latest_shock['ticker']} moved {latest_shock['magnitude']:.2%} at {latest_shock['timestamp']}")
            else:
                logger.warning("No market data fetched in this cycle.")
            
            # 3. Poll News (since last update)
            if provider:
                # Poll last 1 hour of news
                one_hour_ago = (now - datetime.timedelta(hours=1)).strftime('%Y%m%dT%H%M')
                try:
                    news = provider.fetch_news(time_from=one_hour_ago)
                    if not news.empty:
                        logger.info(f"Pulse: Found {len(news)} new headlines in the last hour.")
                        # In a full system, we'd trigger sentiment and TCN re-inference here
                except requests.exceptions.RequestException as e:
                    logger.error(f"Network error polling news: {e}")
            
            if poll_interval_mins == 0:
                logger.info("Single cycle mode complete. Exiting.")
                break

            logger.info(f"Cycle complete. Sleeping for {poll_interval_mins} minutes...")
            time.sleep(poll_interval_mins * 60)
            
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down monitor...")
            break
        except Exception as e:
            logger.critical(f"Unexpected error in monitor loop: {e}", exc_info=True)
            time.sleep(60) # Rest before retry

if __name__ == "__main__":
    # For the first run, we'll just do one cycle instead of an infinite loop
    # In production, this would be a systemd service.
    # We can pass poll interval as an argument
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_intra_day_loop(poll_interval_mins=interval)

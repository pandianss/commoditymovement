import os
import sys
import pandas as pd
import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, COMMODITIES, START_DATE
from data_ingestion.market_data import fetch_yfinance_data
from data_ingestion.news_ingestor import NewsIngestionManager, MockNewsProvider
from data_ingestion.alpha_vantage_provider import AlphaVantageNewsProvider
from news_engine.sentiment_analyzer import process_sentiment
from news_engine.topic_modeler import process_topics
from strategies.inference_engine import run_tcn_inference
from utils.logger import setup_logger

logger = setup_logger("daily_update", log_file="daily_update.log")

def main():
    load_dotenv()
    logger.info("Starting Daily Commodity Update...")
    
    # 1. Update Market Data
    logger.info("Step 1/5: Updating Market Data...")
    try:
        fetch_yfinance_data(COMMODITIES, START_DATE, "commodities_raw.csv") # Now incremental
    except Exception as e:
        logger.error(f"Failed to update market data: {e}")
    
    # 2. Update News Data
    logger.info("Step 2/5: Updating News Data...")
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    news_raw = os.path.join(RAW_DATA_DIR, "news_raw.csv")
    
    if api_key:
        provider = AlphaVantageNewsProvider(api_key)
        logger.info("Using Alpha Vantage for live news.")
    else:
        provider = MockNewsProvider()
        logger.warning("Set ALPHA_VANTAGE_API_KEY for real live data. Using Mock provider.")
        
    manager = NewsIngestionManager(provider)
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    try:
        manager.ingest(today, today, news_raw)
    except Exception as e:
        logger.error(f"Failed to ingest news: {e}")
    
    # 3. Process News Intelligence
    logger.info("Step 3/5: Processing News Intelligence...")
    news_intel = os.path.join(PROCESSED_DATA_DIR, "news_with_intel.csv")
    try:
        df_news = process_sentiment(news_raw, news_intel)
        df_news = process_topics(news_intel, news_intel)
    except Exception as e:
        logger.error(f"Failed to process news intelligence: {e}")
    
    # 4. Feature Store Refresh
    logger.info("Step 4/5: Refreshing Feature Store...")
    # Simulation placeholder
    
    # 5. Continuous Inference
    logger.info("Step 5/5: Generating Live Forecast...")
    store_path = os.path.join(PROCESSED_DATA_DIR, "feature_store_v2.csv")
    model_path = os.path.join(PROCESSED_DATA_DIR, "tcn_gold_model.pth")
    
    if os.path.exists(store_path) and os.path.exists(model_path):
        try:
            df_store = pd.read_csv(store_path, index_col=0, parse_dates=True)
            preds = run_tcn_inference(df_store, "target_GC=F_next_ret", model_path)
            
            latest_pred = preds.iloc[-1]
            logger.info("=== LIVE INTELLIGENCE ALERT (GOLD) ===")
            logger.info(f"Date: {preds.index[-1].date()}")
            logger.info(f"Predicted Return (Median): {latest_pred[0.5]:.4%}")
            logger.info(f"90% Confidence Interval: [{latest_pred[0.05]:.4%}, {latest_pred[0.95]:.4%}]")
            
            if latest_pred[0.5] > 0.01:
                logger.info("ALERT: Bullish signal detected.")
            elif latest_pred[0.5] < -0.01:
                logger.info("ALERT: Bearish signal detected.")
        except Exception as e:
            logger.error(f"Error during inference: {e}")
    else:
        logger.warning("Feature store or model not found. Run previous phases first.")

    logger.info("Daily Update Complete.")

if __name__ == "__main__":
    main()

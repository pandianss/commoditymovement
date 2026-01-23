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

from core.orchestrator import Orchestrator

logger = setup_logger("daily_update", log_file="daily_update.log")

def update_market_data():
    logger.info("Updating Market Data...")
    fetch_yfinance_data(COMMODITIES, START_DATE, "commodities_raw.csv")

def update_news_data():
    logger.info("Updating News Data...")
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    news_raw = os.path.join(RAW_DATA_DIR, "news_raw.csv")
    
    provider = AlphaVantageNewsProvider(api_key) if api_key else MockNewsProvider()
    if not api_key:
        logger.warning("Using Mock provider for news.")
        
    manager = NewsIngestionManager(provider)
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    manager.ingest(today, today, news_raw)

def process_news_intel():
    logger.info("Processing News Intelligence...")
    news_raw = os.path.join(RAW_DATA_DIR, "news_raw.csv")
    news_intel = os.path.join(PROCESSED_DATA_DIR, "news_with_intel.csv")
    process_sentiment(news_raw, news_intel)
    process_topics(news_intel, news_intel)

def refresh_feature_store():
    logger.info("Refreshing Feature Store...")
    # Simulation placeholder
    pass

def generate_forecast():
    logger.info("Generating Live Forecast...")
    store_path = os.path.join(PROCESSED_DATA_DIR, "feature_store_v2.csv")
    model_path = os.path.join(PROCESSED_DATA_DIR, "tcn_gold_model.pth")
    
    if os.path.exists(store_path) and os.path.exists(model_path):
        df_store = pd.read_csv(store_path, index_col=0, parse_dates=True)
        preds = run_tcn_inference(df_store, "target_GC=F_next_ret", model_path)
        
        latest_pred = preds.iloc[-1]
        logger.info(f"GOLD Forecast: Med={latest_pred[0.5]:.4%}")
    else:
        logger.warning("Feature store or model not found.")

def main():
    load_dotenv()
    orchestrator = Orchestrator(state_file="state/daily_update_state.json")
    
    # Register tasks with dependencies
    orchestrator.register_task("market_data", update_market_data)
    orchestrator.register_task("news_data", update_news_data)
    orchestrator.register_task("news_intel", process_news_intel, dependencies=["news_data"])
    orchestrator.register_task("feature_store", refresh_feature_store, dependencies=["market_data"])
    orchestrator.register_task("forecast", generate_forecast, dependencies=["feature_store", "news_intel"])
    
    try:
        orchestrator.run_pipeline("daily_commodity_update")
    except Exception as e:
        logger.critical(f"Daily Update Pipeline Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

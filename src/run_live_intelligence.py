import time
import os
import sys
import datetime
import requests
import torch
import pandas as pd
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, COMMODITIES
from data_ingestion.market_data import fetch_intraday_data
from data_ingestion.alpha_vantage_provider import AlphaVantageNewsProvider
from news_engine.realtime_shocks import detect_intraday_shocks
from utils.logger import setup_logger
from strategies.inference_engine import run_tcn_inference
from models.tcn_engine import TCNQuantileModel  # Needed for loading
from core.state_store import StateStore
from ops.health import HealthMonitor
from ops.drift import DriftDetector
from ops.circuit_breaker import ComponentCircuitBreakers

from core.orchestrator import Orchestrator

logger = setup_logger("live_intelligence", log_file="live_intel.log")

def load_tcn_model():
    model_path = os.path.join(PROCESSED_DATA_DIR, "tcn_gold_model.pth")
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return None, None
        
    store_path = os.path.join(PROCESSED_DATA_DIR, "feature_store.csv")
    if not os.path.exists(store_path):
        logger.error("Feature store not found.")
        return None, None

    df = pd.read_csv(store_path, index_col=0, parse_dates=True)
    feature_cols = [c for c in df.columns if not c.startswith("target_")]
    feat_dim = len(feature_cols)
    
    quantiles = [0.05, 0.5, 0.95]
    model = TCNQuantileModel(input_size=feat_dim, num_channels=[32, 32, 32], 
                             quantiles=quantiles, kernel_size=3)
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logger.info("TCN Model loaded successfully.")
        return model, df
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None

def run_intra_day_loop(poll_interval_mins=30):
    load_dotenv()
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    provider = AlphaVantageNewsProvider(api_key) if api_key else None
    
    orchestrator = Orchestrator(state_file="state/live_intel_state.json")
    model, historical_df = load_tcn_model()
    
    def fetch_prices():
        logger.debug("Fetching 1h intraday data...")
        df_1h = fetch_intraday_data(COMMODITIES, interval="1h", period="5d")
        if df_1h.empty:
            logger.warning("No market data fetched.")
        return df_1h

    def detect_shocks_task():
        df_1h = fetch_prices() # In a real DAG, we'd pass state. For now, simple calls.
        if not df_1h.empty:
            shocks = detect_intraday_shocks(df_1h)
            if not shocks.empty:
                latest_shock = shocks.iloc[-1]
                logger.info(f"!!! SHOCK DETECTED: {latest_shock['ticker']} move at {latest_shock['timestamp']}")

    def poll_news_task():
        if provider:
            now = datetime.datetime.now()
            one_hour_ago = (now - datetime.timedelta(hours=1)).strftime('%Y%m%dT%H%M')
            news = provider.fetch_news(time_from=one_hour_ago)
            if not news.empty:
                logger.info(f"Pulse: Found {len(news)} new headlines.")
                news_path = os.path.join(RAW_DATA_DIR, "live_news_feed.csv")
                mode = 'a' if os.path.exists(news_path) else 'w'
                news.to_csv(news_path, mode=mode, header=(mode=='w'), index=False)

    def run_inference_task():
        if model and historical_df is not None:
            preds = run_tcn_inference(historical_df, "target_GC=F_next_ret", model)
            latest = preds.iloc[-1]
            logger.info(f"PREDICTION: Med={latest[0.5]:.4%}")
            
            # Health & Drift are handled by Orchestrator's internal health monitor
            # or could be added as explicit tasks.

    # Register tasks
    orchestrator.register_task("fetch_prices", fetch_prices)
    orchestrator.register_task("detect_shocks", detect_shocks_task, dependencies=["fetch_prices"])
    orchestrator.register_task("poll_news", poll_news_task)
    orchestrator.register_task("inference", run_inference_task, dependencies=["fetch_prices"])

    logger.info("Starting Intra-day Intelligence Loop...")
    
    while True:
        try:
            # For live loop, we 'force' execution as we want fresh data each cycle
            orchestrator.run_pipeline("live_intelligence_cycle", force=True)
            
            if poll_interval_mins == 0:
                break
            logger.info(f"Cycle complete. Waiting {poll_interval_mins}m...")
            time.sleep(poll_interval_mins * 60)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.critical(f"Loop error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_intra_day_loop(poll_interval_mins=interval)

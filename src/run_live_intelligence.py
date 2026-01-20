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

logger = setup_logger("live_intelligence", log_file="live_intel.log")

def load_tcn_model():
    model_path = os.path.join(PROCESSED_DATA_DIR, "tcn_gold_model.pth")
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return None, 0
        
    # We need to know feat_dim. Ideally saved with model or in config.
    # For now, we load feature store to get dim.
    store_path = os.path.join(PROCESSED_DATA_DIR, "feature_store.csv")
    if not os.path.exists(store_path):
        logger.error("Feature store not found.")
        return None, 0

    df = pd.read_csv(store_path, index_col=0, parse_dates=True)
    # Check feature cols (excluding target_)
    feature_cols = [c for c in df.columns if not c.startswith("target_")]
    feat_dim = len(feature_cols)
    
    quantiles = [0.05, 0.5, 0.95]
    model = TCNQuantileModel(input_size=feat_dim, num_channels=[32, 32, 32], 
                             quantiles=quantiles, kernel_size=3)
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logger.info("TCN Model loaded successfully.")
        return model, df # Return df as 'context' for sequence generation
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, 0

def run_intra_day_loop(poll_interval_mins=30):
    load_dotenv()
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        logger.warning("No ALPHA_VANTAGE_API_KEY found. Intra-day polling will be limited.")
        
    provider = AlphaVantageNewsProvider(api_key) if api_key else None
    
    # Load Model ONCE
    model, historical_df = load_tcn_model()
    
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
                one_hour_ago = (now - datetime.timedelta(hours=1)).strftime('%Y%m%dT%H%M')
                try:
                    news = provider.fetch_news(time_from=one_hour_ago)
                    if not news.empty:
                        logger.info(f"Pulse: Found {len(news)} new headlines.")
                        # Save to CSV for Dashboard
                        news_path = os.path.join(RAW_DATA_DIR, "live_news_feed.csv")
                        mode = 'a' if os.path.exists(news_path) else 'w'
                        header = not os.path.exists(news_path)
                        news.to_csv(news_path, mode=mode, header=header, index=False)
                except requests.exceptions.RequestException as e:
                    logger.error(f"Network error polling news: {e}")
            
            # 4. TCN Inference (Live Prediction)
            if model and historical_df is not None:
                # In a real live setting, we'd append new intraday data to historical_df here.
                # For this baseline integration, we predict on the LATEST available historical day.
                # This confirms the model integration works.
                try:
                    preds = run_tcn_inference(historical_df, "target_GC=F_next_ret", model)
                    latest = preds.iloc[-1]
                    logger.info(f"PREDICTION (Gold 1d): Med={latest[0.5]:.4%} | Range=[{latest[0.05]:.4%}, {latest[0.95]:.4%}]")
                    
                    if latest[0.5] > 0.01:
                        logger.info("SIGNAL: BULLISH")
                    elif latest[0.5] < -0.01:
                        logger.info("SIGNAL: BEARISH")
                    else:
                        logger.info("SIGNAL: NEUTRAL")
                        
                except Exception as e:
                    logger.error(f"Inference failed: {e}")

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
            time.sleep(60)

if __name__ == "__main__":
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_intra_day_loop(poll_interval_mins=interval)

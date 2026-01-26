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
from optimization.optimizer import StrategyOptimizer
from strategies.signal_strategy import ProbabilisticTrendStrategy
from contracts.signal import Signal
import numpy as np
import json

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
    
    # Initialize Orchestrator and operational components
    orchestrator = Orchestrator(state_file="state/live_intel_state.json")
    drift_detector = DriftDetector()
    model, historical_df = load_tcn_model()
    
    def fetch_prices():
        logger.debug("Fetching 1h intraday data...")
        df_1h = fetch_intraday_data(COMMODITIES, interval="1h", period="5d")
        if df_1h.empty:
            logger.warning("No market data fetched.")
        return df_1h

    def detect_shocks_task():
        df_1h = fetch_prices()
        if not df_1h.empty:
            shocks = detect_intraday_shocks(df_1h)
            if not shocks.empty:
                latest_shock = shocks.iloc[-1]
                logger.info(f"!!! SHOCK DETECTED: {latest_shock['ticker']} moved {latest_shock['magnitude']:.2%} at {latest_shock['timestamp']}")

    def poll_news_task():
        if provider:
            now = datetime.datetime.now()
            one_hour_ago = (now - datetime.timedelta(hours=1)).strftime('%Y%m%dT%H%M')
            news = provider.fetch_news(time_from=one_hour_ago)
            if not news.empty:
                logger.info(f"Pulse: Found {len(news)} new headlines.")
                news_path = os.path.join(RAW_DATA_DIR, "live_news_feed.csv")
                mode = 'a' if os.path.exists(news_path) else 'w'
                news.to_csv(news_path, mode=mode, header=(mode=='w' or not os.path.exists(news_path)), index=False)

    def run_inference_task():
        if model and historical_df is not None:
            try:
                preds = run_tcn_inference(historical_df, "target_GC=F_next_ret", model)
                latest = preds.iloc[-1]
                logger.info(f"PREDICTION (Gold 1d): Med={latest[0.5]:.4%} | Range=[{latest[0.05]:.4%}, {latest[0.95]:.4%}]")
                
                # Health Check
                health_status = orchestrator.health_monitor.check_model_health(preds)
                if health_status['status'] != 'healthy':
                    logger.warning(f"Model health degraded: {health_status}")
                
                # Drift Detection
                if drift_detector.baseline is not None:
                    is_drifting, psi = drift_detector.check_drift(preds[0.5])
                    if is_drifting:
                        logger.warning(f"MODEL DRIFT DETECTED! PSI={psi:.4f}")
                else:
                    drift_detector.update_baseline(preds[0.5])
                
                # Save predictions for API and Optimization
                preds_path = os.path.join(PROCESSED_DATA_DIR, "tcn_gold_preds.csv")
                preds.to_csv(preds_path)
                logger.info(f"Saved live predictions to {preds_path}")
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                raise e

    def run_optimization_task():
        # --- OPTIMIZATION BRIDGE ---
        logger.info("Running Live Optimization (Last 180 days)...")
        
        preds_path = os.path.join(PROCESSED_DATA_DIR, "tcn_gold_preds.csv")
        if not os.path.exists(preds_path):
            logger.warning("No predictions available for optimization.")
            return
            
        preds = pd.read_csv(preds_path, index_col=0, parse_dates=True)
        recent_preds = preds.iloc[-180:] 
        
        opt_signals_list = []
        for idx, row in recent_preds.iterrows():
            # CSV columns are strings
            med = row['0.5'] if '0.5' in row else row[0.5]
            direction = 1.0 if med > 0 else -1.0
            prob = min(0.99, 0.5 + abs(med) * 5.0)
            opt_signals_list.append({
                'timestamp_utc': idx,
                'asset': 'GC=F',
                'direction': direction,
                'probability': prob,
                'source': 'TCN_Live'
            })
        
        opt_signals_df = pd.DataFrame(opt_signals_list)
        
        def param_space(trial):
            return {
                'confidence_threshold': trial.suggest_float('confidence_threshold', 0.51, 0.80, step=0.01),
                'max_cap': trial.suggest_float('max_cap', 0.10, 0.50, step=0.05)
            }
        
        opt_data = historical_df.loc[recent_preds.index]
        rec_price = (1 + opt_data.get('target_GC=F_next_ret', pd.Series(0, index=opt_data.index))).cumprod() * 1000
        opt_price_data = pd.DataFrame({'GC=F': rec_price})
        
        optimizer = StrategyOptimizer(ProbabilisticTrendStrategy, opt_price_data, opt_signals_df)
        best_params = optimizer.optimize(param_space, n_trials=10) 
        
        logger.info(f"OPTIMIZED PARAMS: {best_params}")
        
        # Generate Live Signal
        latest_row = preds.iloc[-1]
        latest_med = latest_row['0.5'] if '0.5' in latest_row else latest_row[0.5]
        latest_prob = min(0.99, 0.5 + abs(latest_med) * 5.0)
        latest_dir = 1.0 if latest_med > 0 else -1.0
        
        strategy = ProbabilisticTrendStrategy(**best_params)
        live_sig = Signal(
            timestamp_utc=preds.index[-1],
            asset='GC=F',
            signal_type='LIVE',
            direction=latest_dir,
            probability=latest_prob,
            horizon='1d',
            source='TCN_Live'
        )
        
        alloc = strategy.generate_allocations([live_sig])
        alloc = strategy.apply_risk_budgeting(alloc)
        
        target_weight = 0.0
        if not alloc.empty:
            target_weight = alloc.iloc[0]['weight']
            logger.info(f"*** LIVE ORDER: GOLD Target {target_weight:.2%} (Prob {latest_prob:.2f} vs Thresh {best_params['confidence_threshold']:.2f}) ***")
        else:
            logger.info(f"*** LIVE ORDER: FLAT (Prob {latest_prob:.2f} < Thresh {best_params['confidence_threshold']:.2f}) ***")

        # Save Optimization Result for Dashboard
        live_order_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "signal": "BULLISH" if latest_dir > 0 else "BEARISH",
            "probability": latest_prob,
            "optimized_params": best_params,
            "target_weight": target_weight,
            "status": "ACTIVE" if target_weight != 0 else "WAITING"
        }
        order_path = os.path.join(PROCESSED_DATA_DIR, "live_order.json")
        with open(order_path, 'w') as f:
            json.dump(live_order_data, f, indent=4)

    # Register tasks in the Orchestrator
    orchestrator.register_task("fetch_prices", fetch_prices)
    orchestrator.register_task("detect_shocks", detect_shocks_task, dependencies=["fetch_prices"])
    orchestrator.register_task("poll_news", poll_news_task)
    orchestrator.register_task("inference", run_inference_task, dependencies=["fetch_prices"])
    orchestrator.register_task("optimization", run_optimization_task, dependencies=["inference"])

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

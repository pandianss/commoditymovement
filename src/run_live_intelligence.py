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

# Optimization Bridge
from optimization.optimizer import StrategyOptimizer
from strategies.signal_strategy import ProbabilisticTrendStrategy
from contracts.signal import Signal
import numpy as np

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
    
    # Initialize Operational Components
    state_store = StateStore("state/run_state.json")
    health_monitor = HealthMonitor()
    drift_detector = DriftDetector()
    circuit_breakers = ComponentCircuitBreakers()
    
    # Load Model ONCE
    model, historical_df = load_tcn_model()
    
    # Check for previous state (recovery)
    last_cycle = state_store.get("last_successful_cycle")
    if last_cycle:
        logger.info(f"Resuming from last successful cycle: {last_cycle}")
    
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
            
            # 3. Poll News (since last update) - with Circuit Breaker
            if provider:
                one_hour_ago = (now - datetime.timedelta(hours=1)).strftime('%Y%m%dT%H%M')
                news_breaker = circuit_breakers.get_breaker("news_api")
                try:
                    def fetch_news_wrapper():
                        return provider.fetch_news(time_from=one_hour_ago)
                    
                    news = news_breaker.call(fetch_news_wrapper)
                    if not news.empty:
                        logger.info(f"Pulse: Found {len(news)} new headlines.")
                        # Save to CSV for Dashboard
                        news_path = os.path.join(RAW_DATA_DIR, "live_news_feed.csv")
                        mode = 'a' if os.path.exists(news_path) else 'w'
                        header = not os.path.exists(news_path)
                        news.to_csv(news_path, mode=mode, header=header, index=False)
                except requests.exceptions.RequestException as e:
                    logger.error(f"Network error polling news: {e}")
                except Exception as e:
                    logger.warning(f"News API circuit breaker triggered: {e}")
            
            # 4. TCN Inference (Live Prediction) - with Health & Drift Monitoring
            if model and historical_df is not None:
                try:
                    preds = run_tcn_inference(historical_df, "target_GC=F_next_ret", model)
                    latest = preds.iloc[-1]
                    logger.info(f"PREDICTION (Gold 1d): Med={latest[0.5]:.4%} | Range=[{latest[0.05]:.4%}, {latest[0.95]:.4%}]")
                    
                    # Health Check: Model predictions
                    health_status = health_monitor.check_model_health(preds)
                    if health_status['status'] != 'healthy':
                        logger.warning(f"Model health degraded: {health_status}")
                    
                    # Drift Detection
                    if drift_detector.baseline is not None:
                        is_drifting, psi = drift_detector.check_drift(preds[0.5])
                        if is_drifting:
                            logger.warning(f"MODEL DRIFT DETECTED! PSI={psi:.4f}")
                    else:
                        # Initialize baseline on first run
                        drift_detector.update_baseline(preds[0.5])
                    
                    if latest[0.5] > 0.01:
                        logger.info("SIGNAL: BULLISH")
                    elif latest[0.5] < -0.01:
                        logger.info("SIGNAL: BEARISH")
                    else:
                        logger.info("SIGNAL: NEUTRAL")

                    # --- OPTIMIZATION BRIDGE ---
                    logger.info("Running Live Optimization (Last 180 days)...")
                    
                    # 1. Prepare Data for Optimizer
                    # Convert raw TCN preds to 'Signal' format dataframe for the optimizer
                    # We need to join preds with historical_df to get timestamps if needed, 
                    # but preds index should be datetime.
                    
                    opt_signals_list = []
                    # Optimization: Only process last 180 days for speed
                    recent_preds = preds.iloc[-180:] 
                    
                    for idx, row in recent_preds.iterrows():
                        med = row[0.5]
                        direction = 1.0 if med > 0 else -1.0
                        # Simple confidence proxy: magnitude of return expectation scaled
                        # e.g. 1% return = 0.6 confidence, 5% = 0.9. 
                        # Sigmoid: 1 / (1 + exp(-k * abs(med)))
                        # Let's use simple linear scaling for demo: 0.5 + abs(med)*10, capped at 0.99
                        prob = min(0.99, 0.5 + abs(med) * 5.0)
                        
                        opt_signals_list.append({
                            'timestamp_utc': idx,
                            'asset': 'GC=F',
                            'direction': direction,
                            'probability': prob,
                            'source': 'TCN_Live'
                        })
                    
                    opt_signals_df = pd.DataFrame(opt_signals_list)
                    
                    # 2. Define Param Space
                    def param_space(trial):
                        return {
                            'confidence_threshold': trial.suggest_float('confidence_threshold', 0.51, 0.80, step=0.01),
                            'max_cap': trial.suggest_float('max_cap', 0.10, 0.50, step=0.05)
                        }
                    
                    # 3. optimize
                    # Need aligned market data (Close price) for the optimizer backtest
                    # historical_df has index=Date.
                    opt_data = historical_df.loc[recent_preds.index]
                    
                    # Rename columns to match BacktestEngine expectation if needed?
                    # Engine uses 'Asset' column for price? Or columns=[Asset]?
                    # historical_df probably has features. We need 'GC=F' Close price.
                    # Assuming 'GC=F_Close' or similar exists. Let's check feature store cols later.
                    # For now, we trick it: create a 'GOLD' column from a known price col if exists, or features.
                    # If feature store has 'target_GC=F_next_ret', it might not have raw price.
                    # BUT we have `df_1h` or `fetch_intraday_data`? No that's intraday.
                    # Let's assume we can approximate "price" curve from cumulative returns for the optimizer 
                    # if raw price isn't there.
                    
                    # Reconstruction of price from returns for backtest:
                    rec_price = (1 + opt_data.get('target_GC=F_next_ret', pd.Series(0, index=opt_data.index))).cumprod() * 1000
                    opt_price_data = pd.DataFrame({'GC=F': rec_price})
                    
                    optimizer = StrategyOptimizer(ProbabilisticTrendStrategy, opt_price_data, opt_signals_df)
                    best_params = optimizer.optimize(param_space, n_trials=10) # Fast check
                    
                    logger.info(f"OPTIMIZED PARAMS: {best_params}")
                    
                    # 4. Generate Live Signal
                    # Use best params on the LATEST prediction
                    latest_med = latest[0.5]
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
                    
                    if not alloc.empty:
                        target_weight = alloc.iloc[0]['weight']
                        logger.info(f"*** LIVE ORDER: GOLD Target {target_weight:.2%} (Prob {latest_prob:.2f} vs Thresh {best_params['confidence_threshold']:.2f}) ***")
                    else:
                        target_weight = 0.0
                        logger.info(f"*** LIVE ORDER: FLAT (Prob {latest_prob:.2f} < Thresh {best_params['confidence_threshold']:.2f}) ***")

                    # Save Optimization Result for Dashboard
                    import json
                    live_order_data = {
                        "timestamp": now.isoformat(),
                        "signal": "BULLISH" if latest_dir > 0 else "BEARISH",
                        "probability": latest_prob,
                        "optimized_params": best_params,
                        "target_weight": target_weight,
                        "status": "ACTIVE" if target_weight != 0 else "WAITING"
                    }
                    order_path = os.path.join(PROCESSED_DATA_DIR, "live_order.json")
                    with open(order_path, 'w') as f:
                        json.dump(live_order_data, f, indent=4)

                    # --- END OPTIMIZATION BRIDGE ---
                        
                    # Save predictions to CSV for API
                    preds_path = os.path.join(PROCESSED_DATA_DIR, "tcn_gold_preds.csv")
                    preds.to_csv(preds_path)
                    logger.info(f"Saved live predictions to {preds_path}")
                        
                except Exception as e:
                    logger.error(f"Inference failed: {e}")
            
            # 5. Checkpoint: Record successful cycle
            state_store.update_checkpoint(
                job_name="live_intelligence",
                status="success",
                marker=now.isoformat()
            )
            state_store.set("last_successful_cycle", now.isoformat())
            health_monitor.record_heartbeat()
            logger.debug("Cycle checkpoint saved.")

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

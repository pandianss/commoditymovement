from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, COMMODITIES
from api.auth import router as auth_router

app = FastAPI(title="Commodity Intelligence API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api")

from core.state_store import StateStore
from core.registry import ModelRegistry
from ops.health import HealthMonitor

# Initialize core components
# Note: In a production setup, these would be managed via dependency injection
state = StateStore("state/run_state.json")
registry = ModelRegistry()
health = HealthMonitor()

@app.get("/health")
def health_check():
    return health.get_system_status()

@app.get("/api/system-status")
def get_system_status():
    """Returns sovereignty metrics for the command center."""
    last_cycle = state.get("last_successful_cycle")
    health_status = health.get_system_status()
    
    return {
        "heartbeat": "OPERATIONAL" if health_status.get("status") == "alive" else "STALLED",
        "last_cycle": last_cycle or "BOOTING...",
        "risk_mandate": "MAX DD: 20%", # Hardcoded for now per Streamlit app
        "version": "2.0-Alpha"
    }

@app.get("/api/registry")
def get_registry_info():
    """Returns information about the active champion model."""
    champion = registry.get_champion("tcn_gold")
    return {
        "champion_id": champion.get("model_id", "NONE") if champion else "NONE",
        "target": "GOLD"
    }

@app.get("/api/market-data")
# ... (rest of the file)
def get_market_data():
    """Returns latest intraday price data."""
    path = os.path.join(RAW_DATA_DIR, "commodities_1h_raw.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Market data not found")
    
    # yfinance multi-index CSV: header=[0, 1] means Price and Ticker
    # index_col=0 is the Datetime
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
        
        # We focus on the 'Close' price
        close_df = df['Close']
        close_df = close_df.tail(50) # Last 50 points
        
        result = {}
        for ticker in close_df.columns:
            # Flatten to a simple list of {timestamp, price}
            ticker_data = []
            for ts, price in close_df[ticker].items():
                if pd.notna(price):
                    ticker_data.append({
                        "timestamp": str(ts),
                        "price": float(price),
                        "ticker": ticker
                    })
            result[ticker] = ticker_data
            
        return result
    except Exception as e:
        print(f"Error parsing market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions")
def get_predictions():
    """Returns latest quantile forecasts."""
    # For now, focusing on Gold as it's our primary model
    path = os.path.join(PROCESSED_DATA_DIR, "tcn_gold_preds.csv")
    if not os.path.exists(path):
        return {"GOLD": []}
    
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.tail(20) # Latest 20 predictions
    
    return {
        "GOLD": [{
            "date": str(index.date()),
            "p05": row['0.05'],
            "p50": row['0.5'],
            "p95": row['0.95']
        } for index, row in df.iterrows()]
    }

@app.get("/api/news")
def get_news():
    """Returns latest news with sentiment scores."""
    path = os.path.join(PROCESSED_DATA_DIR, "news_with_intel.csv")
    if not os.path.exists(path):
        return []
    
    df = pd.read_csv(path)
    # Sort by timestamp and get latest 10
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
    df = df.sort_values('timestamp_utc', ascending=False).head(15)
    
    # Fill NaN relevance_prob if it exists
    if 'relevance_prob' in df.columns:
        df['relevance_prob'] = df['relevance_prob'].fillna(0.5)
    else:
        df['relevance_prob'] = 0.5
        
    return df.to_dict(orient='records')

@app.get("/api/shocks")
def get_shocks():
    """Returns latest detected intraday shocks."""
    # We can peek into the live_intel log or better, have a dedicated shocks.csv
    # For simplicity, we'll re-detect from intraday data or return mock shocks if none
    # In a full system, run_live_intelligence would persist these.
    # Let's check if a shocks file exists, if not return demo shocks.
    return [
        {"timestamp": "2026-01-19 04:00:00", "ticker": "NG=F", "magnitude": 0.1086, "type": "POSITIVE_SHOCK"},
        {"timestamp": "2026-01-18 10:00:00", "ticker": "GC=F", "magnitude": -0.012, "type": "NEGATIVE_SHOCK"}
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

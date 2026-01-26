import os
from dotenv import load_dotenv

load_dotenv()

# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Time Horizon
START_DATE = "2010-01-01"
# End date is usually 'today', but for consistency:
TRAIN_END_DATE = "2024-12-31"

# Target Commodities (Yahoo Finance Tickers)
from utils.ticker_mapper import TickerMapper

# Initialize Mapper
try:
    _mapper = TickerMapper(os.path.join(os.path.dirname(os.path.dirname(__file__)), "etc", "ticker_map.json"))
    _assets = _mapper.get_all_assets()
    # YFinance Tickers for Data Ingestion
    COMMODITIES = {k: v['yfinance'] for k, v in _assets.items()}
    # Full Asset Metadata
    ASSET_UNIVERSE = _assets
except Exception as e:
    print(f"Warning: Could not load ticker map: {e}")
    COMMODITIES = {}
    ASSET_UNIVERSE = {}

# Macro Indicators
MACRO_DRIVERS = {
    "DXY": "DX-Y.NYB",       # US Dollar Index
    "UST_10Y": "^TNX",       # 10Y Treasury Yield
    "UST_2Y": "^IRX",        # 13-week T-Bill (proxy for 2Y if needed, but ^IRX is 13w)
    "VIX": "^VIX",           # Volatility Index
    "SP500": "^GSPC",        # S&P 500
}

# Derived features lag windows
LAG_WINDOWS = [1, 5, 20] # 1 day, 1 week, 1 month (approx)

# Backtest Configuration
WALK_FORWARD_CONFIG = {
    "start_year": 2011,
    "train_size_years": 8, # 2011-2018 for first test year 2019
    "test_size_years": 1,
}

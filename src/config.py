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
COMMODITIES = {
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "CRUDE_OIL": "CL=F",
    "COPPER": "HG=F",
    "NATURAL_GAS": "NG=F",
}

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

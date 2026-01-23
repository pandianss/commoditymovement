import yfinance as yf
import pandas as pd
import os
import sys
import time

# Add src to path if needed for local execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COMMODITIES, MACRO_DRIVERS, RAW_DATA_DIR, START_DATE, ASSET_UNIVERSE

import datetime

def fetch_yfinance_data(tickers, start_date, output_filename):
    """
    Fetches daily OHLCV data incrementally if file exists, otherwise full download.
    Optimized for batch downloading.
    """
    output_path = os.path.join(RAW_DATA_DIR, output_filename)
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    ticker_list = list(tickers.values())
    
    if not ticker_list:
        print(f"No tickers found for {output_filename}, skipping.")
        return

    print(f"Processing ingestion for {len(ticker_list)} assets into {output_filename}...")
    
    if os.path.exists(output_path):
        try:
            # Check last date
            existing_df = pd.read_csv(output_path, index_col=0, header=[0, 1], parse_dates=True)
            if not existing_df.empty:
                last_date = existing_df.index.max()
                download_start = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                
                if download_start >= today:
                    print(f"Data up to date.")
                    return existing_df
                
                print(f"Fetching delta: {download_start} -> {today}")
                try:
                    new_data = yf.download(ticker_list, start=download_start, end=today, progress=False)
                    if not new_data.empty:
                        new_data.to_csv(output_path, mode='a', header=False)
                        print(f"Appended {len(new_data)} rows.")
                    return
                except Exception as e:
                    print(f"Incremental update failed: {e}. Triggering full refresh.")
        except Exception as e:
            print(f"File corruption detected ({e}). Triggering full refresh.")

    # Full Download
    print(f"Downloading full history from {start_date}...")
    try:
        data = yf.download(ticker_list, start=start_date, end=today, group_by='ticker', progress=False)
        # YFinance structure changes based on 1 vs Many tickers. We enforce multi-index.
        if len(ticker_list) == 1:
            # Reformat single ticker DF to match multi-index structure of batch
            data.columns = pd.MultiIndex.from_product([data.columns, ticker_list])
            
        data.to_csv(output_path)
        print(f"Success. Saved to {output_path}")
        return data
    except Exception as e:
        print(f"Download failed: {e}")

def fetch_intraday_data(tickers, interval="1h", period="60d"):
    """
    Fetches intra-day data.
    """
    output_filename = f"assets_{interval}_raw.csv"
    output_path = os.path.join(RAW_DATA_DIR, output_filename)
    ticker_list = list(tickers.values())
    
    print(f"Fetching {interval} intraday for {len(ticker_list)} assets...")
    try:
        data = yf.download(ticker_list, period=period, interval=interval, group_by='ticker', progress=False)
        data.to_csv(output_path)
        print(f"Saved to {output_path}")
    except Exception as e:
        print(f"Intraday fetch failed: {e}")

def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # 1. Daily Updates (Universal Asset Batch)
    # Merging Commodities, Equities, Indices into one master file for simplicity
    # or separating if volume is huge. For now, unified.
    fetch_yfinance_data(COMMODITIES, START_DATE, "commodities_raw.csv") # Keeping legacy name for now to avoid breaking pipeline
    fetch_yfinance_data(MACRO_DRIVERS, START_DATE, "macro_raw.csv")

    # 2. Intra-day Updates
    fetch_intraday_data(COMMODITIES, interval="1h", period="60d")
    
    # Optional: 15m data for active assets only (filtering logic could go here)
    fetch_intraday_data(COMMODITIES, interval="15m", period="60d")

if __name__ == "__main__":
    main()

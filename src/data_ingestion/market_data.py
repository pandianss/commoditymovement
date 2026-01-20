import yfinance as yf
import pandas as pd
import os
import sys

# Add src to path if needed for local execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COMMODITIES, MACRO_DRIVERS, RAW_DATA_DIR, START_DATE

import datetime

def fetch_yfinance_data(tickers, start_date, output_filename):
    """
    Fetches daily OHLCV data incrementally if file exists, otherwise full download.
    """
    output_path = os.path.join(RAW_DATA_DIR, output_filename)
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    if os.path.exists(output_path):
        # We use a helper to find the last date safely
        try:
            existing_df = pd.read_csv(output_path, index_col=0, header=[0, 1], parse_dates=True)
            last_date = existing_df.index.max()
            
            # Ensure last_date is a timestamp and add 1 day
            download_start = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
            if download_start >= today:
                print(f"Data in {output_filename} is up to date (Last date: {last_date.date()}).")
                return existing_df
                
            print(f"Fetching incremental data for {output_filename} from {download_start} to {today}...")
            new_data = yf.download(list(tickers.values()), start=download_start, end=today)
            
            if not new_data.empty:
                # Reload full to concatenate correctly with multi-index headers if needed
                # For simplicity in this script, we'll just append
                new_data.to_csv(output_path, mode='a', header=False)
                print(f"Appended {len(new_data)} rows to {output_filename}")
                return pd.read_csv(output_path, index_col=0, parse_dates=True)
            return existing_df
            
        except Exception as e:
            print(f"Error reading existing file {output_filename}: {e}. Falling back to full download.")
            
    print(f"Running full download for {output_filename} from {start_date} to {today}...")
    data = yf.download(list(tickers.values()), start=start_date, end=today)
    data.to_csv(output_path)
    print(f"Saved raw data to {output_path}")
    return data

def fetch_intraday_data(tickers, interval="1h", period="60d"):
    """
    Fetches intra-day data for a given interval.
    Note: yfinance allows max 7d for 1m, 60d for 1h/15m.
    """
    print(f"Fetching {interval} intra-day data for: {list(tickers.keys())}...")
    data = yf.download(list(tickers.values()), period=period, interval=interval)
    
    output_filename = f"commodities_{interval}_raw.csv"
    output_path = os.path.join(RAW_DATA_DIR, output_filename)
    data.to_csv(output_path)
    print(f"Saved intra-day data to {output_path}")
    return data

def main():
    # Ensure raw data dir exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # 1. Daily Updates (Historical Baseline)
    fetch_yfinance_data(COMMODITIES, START_DATE, "commodities_raw.csv")
    fetch_yfinance_data(MACRO_DRIVERS, START_DATE, "macro_raw.csv")

    # 2. Intra-day Updates (Volatility Monitoring)
    print("\n--- Starting High-Frequency Ingestion ---")
    fetch_intraday_data(COMMODITIES, interval="1h", period="60d")
    fetch_intraday_data(COMMODITIES, interval="15m", period="60d")

if __name__ == "__main__":
    main()

import yfinance as yf
import pandas as pd
import os
import sys
import time
import datetime

# Add src to path if needed for local execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COMMODITIES, MACRO_DRIVERS, RAW_DATA_DIR, START_DATE, ASSET_UNIVERSE

def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def fetch_yfinance_data(tickers, start_date, output_filename, chunk_size=50):
    """
    Fetches daily OHLCV data incrementally if file exists, otherwise full download.
    Optimized for batch downloading with chunking to avoid throttling.
    """
    output_path = os.path.join(RAW_DATA_DIR, output_filename)
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    ticker_list = list(tickers.values())
    
    if not ticker_list:
        print(f"No tickers found for {output_filename}, skipping.")
        return
    
    print(f"Processing ingestion for {len(ticker_list)} assets into {output_filename} (Chunk Size: {chunk_size})...")
    
    all_data = []
    
    # Simple strategy: Full download always for now if the universe is large, 
    # or implement robust merging. Given the requirement for scalability, 
    # we'll use yf.download on chunks and combine.
    
    for chunk in chunk_list(ticker_list, chunk_size):
        print(f"Downloading chunk: {chunk[:3]}... ({len(chunk)} assets)")
        try:
            chunk_data = yf.download(chunk, start=start_date, end=today, group_by='ticker', progress=False)
            if not chunk_data.empty:
                # If only one ticker in chunk, yfinance returns a single-level column DF
                if len(chunk) == 1:
                    chunk_data.columns = pd.MultiIndex.from_product([chunk, chunk_data.columns]).swaplevel()
                all_data.append(chunk_data)
            time.sleep(1) # Subtle delay to be polite
        except Exception as e:
            print(f"Failed to download chunk {chunk[:3]}: {e}")

    if all_data:
        final_df = pd.concat(all_data, axis=1)
        final_df.to_csv(output_path)
        print(f"Success. Saved {len(final_df)} rows for {len(ticker_list)} assets to {output_path}")
        return final_df
    
    return None

def fetch_intraday_data(tickers, interval="1h", period="60d", chunk_size=50):
    """
    Fetches intra-day data with chunking.
    """
    output_filename = f"assets_{interval}_raw.csv"
    output_path = os.path.join(RAW_DATA_DIR, output_filename)
    ticker_list = list(tickers.values())
    
    print(f"Fetching {interval} intraday for {len(ticker_list)} assets (Chunk Size: {chunk_size})...")
    
    all_data = []
    for chunk in chunk_list(ticker_list, chunk_size):
        print(f"Downloading {interval} chunk: {chunk[:3]}...")
        try:
            chunk_data = yf.download(chunk, period=period, interval=interval, group_by='ticker', progress=False)
            if not chunk_data.empty:
                if len(chunk) == 1:
                    chunk_data.columns = pd.MultiIndex.from_product([chunk, chunk_data.columns]).swaplevel()
                all_data.append(chunk_data)
            time.sleep(1)
        except Exception as e:
            print(f"Failed intraday chunk {chunk[:3]}: {e}")

    if all_data:
        final_df = pd.concat(all_data, axis=1)
        final_df.to_csv(output_path)
        print(f"Saved to {output_path}")
        return final_df
    
    return None

def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Use the expanded ASSET_UNIVERSE
    yfinance_universe = {k: v['yfinance'] for k, v in ASSET_UNIVERSE.items()}
    
    # 1. Daily Updates (Unified Universe)
    fetch_yfinance_data(yfinance_universe, START_DATE, "commodities_raw.csv") # Unified file
    fetch_yfinance_data(MACRO_DRIVERS, START_DATE, "macro_raw.csv")

    # 2. Intra-day Updates (Unified Universe)
    fetch_intraday_data(yfinance_universe, interval="1h", period="60d")
    
    # Optional: 15m data for the whole universe might be too much, but let's stick to the script
    fetch_intraday_data(yfinance_universe, interval="15m", period="60d")

if __name__ == "__main__":
    main()

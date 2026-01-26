import requests
import pandas as pd
import os
import sys
import argparse
import time
from datetime import datetime, timedelta
import io

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("bse_fetcher", log_file="bse_fetcher.log")

class BSEHistoricalFetcher:
    """
    Fetches historical equity data from BSE's daily Bhavcopy archives.
    URL Format: https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{YYYYMMDD}_F_0000.CSV
    """
    
    BASE_URL = "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{}_F_0000.CSV"
    
    def __init__(self, output_dir="data/raw"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = os.path.join(self.output_dir, "bse_history_raw.csv")

    def fetch_data(self, start_date, end_date):
        """
        Iterates through the date range and downloads daily Bhavcopy files.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        delta = timedelta(days=1)
        
        all_data = []
        
        current_date = start
        while current_date <= end:
            # Skip weekends
            if current_date.weekday() >= 5: # 5=Sat, 6=Sun
                current_date += delta
                continue
                
            date_str = current_date.strftime("%Y%m%d")
            url = self.BASE_URL.format(date_str)
            
            logger.info(f"Fetching BSE data for {current_date.strftime('%Y-%m-%d')}...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    df = pd.read_csv(io.StringIO(response.text))
                    
                    # Standardize columns
                    # BSE 2024 Format: TckrSymb, FinInstrmNm, OpnPric, HghPric, LwPric, ClsPric, TtlTradgVol
                    col_map = {
                        'TckrSymb': 'SC_CODE', # Using Symbol as Code for this format, or keep as is
                        'FinInstrmNm': 'SC_NAME',
                        'OpnPric': 'OPEN',
                        'HghPric': 'HIGH',
                        'LwPric': 'LOW', 
                        'ClsPric': 'CLOSE',
                        'TtlTradgVol': 'NO_OF_SHRS'
                    }
                    
                    # Check if new format exists
                    if 'ClsPric' in df.columns:
                        df = df.rename(columns=col_map)
                        
                    # Handle Legacy Format if needed (SC_CODE, CLOSE)
                    if 'SC_CODE' in df.columns and 'CLOSE' in df.columns:
                        # Add Date column
                        df['DATE'] = current_date.strftime("%Y-%m-%d")
                        
                        # Filter only necessary columns
                        required_cols = ['SC_CODE', 'SC_NAME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'NO_OF_SHRS', 'DATE']
                        df = df[required_cols]
                        
                        all_data.append(df)
                        logger.info(f"Successfully fetched {len(df)} records.")
                    else:
                        logger.warning(f"Unexpected columns for {date_str}: {df.columns}")
                        
                elif response.status_code == 404:
                    logger.info(f"Data not found for {date_str} (Likely Market Holiday).")
                else:
                    logger.warning(f"Failed to fetch {date_str}: HTTP {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error fetching {date_str}: {e}")
            
            current_date += delta
            time.sleep(0.5) # Polite delay
            
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            self._save_data(final_df)
            return final_df
        else:
            logger.warning("No data fetched for the given range.")
            return pd.DataFrame()

    def fetch_latest(self):
        """Fetches data for the current date (and previous few days to be safe)."""
        today = datetime.now()
        start_date = (today - timedelta(days=5)).strftime("%Y-%m-%d") # Go back 5 days to cover weekends/holidays
        end_date = today.strftime("%Y-%m-%d")
        logger.info(f"Fetching latest data from {start_date} to {end_date}...")
        return self.fetch_data(start_date, end_date)

    def _save_data(self, new_df):
        """Saves data to CSV, appending if exists."""
        if os.path.exists(self.output_file):
            # Read existing to avoid duplicates if needed, or just append
            # For simplicity in this 'raw' cull, we'll overwrite or append carefully.
            # Let's just overwrite properly for this cull session or append?
            # User asked to "cull relevant historical data". 
            # Safest is to append new dates, but loading entire history to dedup is heavy.
            # We'll validly check last date if we were doing incremental, but here we just save what we fetched.
            
            # Use append mode with header check
            header = False
            new_df.to_csv(self.output_file, mode='a', header=header, index=False)
            logger.info(f"Appended {len(new_df)} records to {self.output_file}")
        else:
            new_df.to_csv(self.output_file, index=False)
            logger.info(f"Saved {len(new_df)} records to {self.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch BSE Historical Data')
    parser.add_argument('--start_date', type=str, required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, required=True, help='End date YYYY-MM-DD')
    
    args = parser.parse_args()
    
    fetcher = BSEHistoricalFetcher()
    fetcher.fetch_data(args.start_date, args.end_date)

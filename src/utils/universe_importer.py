import pandas as pd
import requests
import io
import os
import json
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("universe_importer", log_file="importer.log")

class ExchangeImporter:
    """
    Automates the ingestion of full exchange lists (NSE/BSE)
    to move beyond hardcoded ticker maps.
    """
    
    NSE_MASTER_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    NIFTY_500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

    def __init__(self, target_file="etc/ticker_map.json"):
        self.target_file = target_file

    def _get_sector_map(self, industry_name):
        """Maps NSE Industry to internal simplified sectors."""
        mapping = {
            "Financial Services": "BFSI",
            "Information Technology": "IT",
            "Oil Gas & Consumable Fuels": "ENERGY",
            "Fast Moving Consumer Goods": "FMCG",
            "Automobile and Auto Components": "AUTO",
            "Healthcare": "PHARMA",
            "Chemicals": "CHEMICALS",
            "Consumer Durables": "CONSUMER",
            "Construction": "CONSTRUCTION",
            "Metals & Mining": "METALS",
            "Telecommunication": "TELECOM",
            "Power": "UTILITIES"
        }
        return mapping.get(str(industry_name), "OTHERS")

    def import_from_nse(self, mode="NIFTY_500"):
        """
        Downloads exchange lists and converts to ticker_map format.
        Modes: 'NIFTY_500' (High Quality) or 'FULL' (All NSE Equities)
        """
        logger.info(f"Fetching NSE list (Mode: {mode})...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            universe = {}
            # Preserve existing map
            if os.path.exists(self.target_file):
                with open(self.target_file, 'r') as f:
                    universe = json.load(f)
            
            if mode == "NIFTY_500":
                resp = requests.get(self.NIFTY_500_URL, headers=headers)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text))
                for _, row in df.iterrows():
                    symbol = row['Symbol']
                    universe[symbol] = {
                        "yfinance": f"{symbol}.NS",
                        "kite": symbol,
                        "type": "EQUITY",
                        "sector": self._get_sector_map(row['Industry'])
                    }
            else:
                resp = requests.get(self.NSE_MASTER_URL, headers=headers)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text))
                for _, row in df.iterrows():
                    symbol = row['SYMBOL']
                    if symbol not in universe: # Avoid overwriting curated data
                        universe[symbol] = {
                            "yfinance": f"{symbol}.NS",
                            "kite": symbol,
                            "type": "EQUITY",
                            "sector": "OTHERS"
                        }

            # Save
            os.makedirs(os.path.dirname(self.target_file), exist_ok=True)
            with open(self.target_file, 'w') as f:
                json.dump(universe, f, indent=4)
            
            logger.info(f"Successfully updated universe. Total assets: {len(universe)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import NSE universe: {e}")
            return False

    def import_from_bse(self):
        """
        Import BSE listed companies.
        Note: BSE doesn't have a public CSV like NSE, so we use a curated list approach.
        For yfinance, BSE tickers use .BO suffix.
        """
        logger.info("Importing BSE constituents...")
        
        # BSE Sensex 30 constituents (as a starting point)
        # In production, you'd fetch from a more comprehensive source
        bse_sensex = {
            "RELIANCE": {"name": "Reliance Industries", "sector": "ENERGY"},
            "TCS": {"name": "Tata Consultancy Services", "sector": "IT"},
            "HDFCBANK": {"name": "HDFC Bank", "sector": "BFSI"},
            "INFY": {"name": "Infosys", "sector": "IT"},
            "ICICIBANK": {"name": "ICICI Bank", "sector": "BFSI"},
            "HINDUNILVR": {"name": "Hindustan Unilever", "sector": "FMCG"},
            "ITC": {"name": "ITC Limited", "sector": "FMCG"},
            "SBIN": {"name": "State Bank of India", "sector": "BFSI"},
            "BHARTIARTL": {"name": "Bharti Airtel", "sector": "TELECOM"},
            "KOTAKBANK": {"name": "Kotak Mahindra Bank", "sector": "BFSI"},
            "LT": {"name": "Larsen & Toubro", "sector": "CONSTRUCTION"},
            "AXISBANK": {"name": "Axis Bank", "sector": "BFSI"},
            "ASIANPAINT": {"name": "Asian Paints", "sector": "CONSUMER"},
            "MARUTI": {"name": "Maruti Suzuki", "sector": "AUTO"},
            "SUNPHARMA": {"name": "Sun Pharmaceutical", "sector": "PHARMA"},
            "TITAN": {"name": "Titan Company", "sector": "CONSUMER"},
            "BAJFINANCE": {"name": "Bajaj Finance", "sector": "BFSI"},
            "ULTRACEMCO": {"name": "UltraTech Cement", "sector": "CONSTRUCTION"},
            "NESTLEIND": {"name": "Nestle India", "sector": "FMCG"},
            "WIPRO": {"name": "Wipro", "sector": "IT"}
        }
        
        try:
            universe = {}
            # Preserve existing map
            if os.path.exists(self.target_file):
                with open(self.target_file, 'r') as f:
                    universe = json.load(f)
            
            # Add BSE tickers
            for symbol, meta in bse_sensex.items():
                # Create BSE-specific key if not already present
                bse_key = f"{symbol}_BSE"
                if bse_key not in universe:
                    universe[bse_key] = {
                        "yfinance": f"{symbol}.BO",
                        "kite": symbol,
                        "type": "EQUITY",
                        "sector": meta["sector"],
                        "exchange": "BSE"
                    }
            
            # Save
            os.makedirs(os.path.dirname(self.target_file), exist_ok=True)
            with open(self.target_file, 'w') as f:
                json.dump(universe, f, indent=4)
            
            logger.info(f"Successfully added BSE constituents. Total assets: {len(universe)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import BSE universe: {e}")
            return False

if __name__ == "__main__":
    importer = ExchangeImporter()
    # By default, import Nifty 500 for quality
    importer.import_from_nse(mode="NIFTY_500")

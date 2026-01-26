import json
import os

class TickerMapper:
    """
    Maps standard internal IDs to provider-specific tickers.
    
    Structure:
    {
        "internal_id": "RELIANCE",
        "yfinance": "RELIANCE.NS",
        "kite": "RELIANCE",
        "name": "Reliance Industries",
        "type": "EQUITY"
    }
    """
    
    def __init__(self, mapping_file="etc/ticker_map.json"):
        self.mapping_file = mapping_file
        self.mappings = self._load_mappings()

    def _load_mappings(self):
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, "r") as f:
                return json.load(f)
        return {}

    def refresh_from_file(self):
        """Reloads the mappings from the JSON file."""
        self.mappings = self._load_mappings()
        return len(self.mappings)

    def get_yfinance_ticker(self, internal_id):
        return self.mappings.get(internal_id, {}).get("yfinance")

    def get_kite_ticker(self, internal_id):
        return self.mappings.get(internal_id, {}).get("kite")

    def get_all_assets(self):
        return self.mappings

# Default Universe for Initialization
DEFAULT_UNIVERSE = {
    "GOLD": {"yfinance": "GC=F", "kite": "GOLDM", "type": "COMMODITY", "sector": "PRECIOUS_METALS"},
    "SILVER": {"yfinance": "SI=F", "kite": "SILVERM", "type": "COMMODITY", "sector": "PRECIOUS_METALS"},
    "CRUDE_OIL": {"yfinance": "CL=F", "kite": "CRUDEOIL", "type": "COMMODITY", "sector": "ENERGY"},
    "NIFTY_50": {"yfinance": "^NSEI", "kite": "NIFTY 50", "type": "INDEX", "sector": "BENCHMARK"},
    "BANK_NIFTY": {"yfinance": "^NSEBANK", "kite": "NIFTY BANK", "type": "INDEX", "sector": "BFSI"},
    "RELIANCE": {"yfinance": "RELIANCE.NS", "kite": "RELIANCE", "type": "EQUITY", "sector": "ENERGY"},
    "HDFCBANK": {"yfinance": "HDFCBANK.NS", "kite": "HDFCBANK", "type": "EQUITY", "sector": "BFSI"},
    "ICICIBANK": {"yfinance": "ICICIBANK.NS", "kite": "ICICIBANK", "type": "EQUITY", "sector": "BFSI"},
    "INFY": {"yfinance": "INFY.NS", "kite": "INFY", "type": "EQUITY", "sector": "IT"},
    "TCS": {"yfinance": "TCS.NS", "kite": "TCS", "type": "EQUITY", "sector": "IT"},
    "ITC": {"yfinance": "ITC.NS", "kite": "ITC", "type": "EQUITY", "sector": "FMCG"},
    "SBIN": {"yfinance": "SBIN.NS", "kite": "SBIN", "type": "EQUITY", "sector": "BFSI"},
    "BHARTIARTL": {"yfinance": "BHARTIARTL.NS", "kite": "BHARTIARTL", "type": "EQUITY", "sector": "TELECOM"},
    "KOTAKBANK": {"yfinance": "KOTAKBANK.NS", "kite": "KOTAKBANK", "type": "EQUITY", "sector": "BFSI"},
    "LT": {"yfinance": "LT.NS", "kite": "LT", "type": "EQUITY", "sector": "CONSTRUCTION"},
    "ZOMATO": {"yfinance": "ZOMATO.NS", "kite": "ZOMATO", "type": "EQUITY", "sector": "NEW_AGE_TECH"},
    "PAYTM": {"yfinance": "PAYTM.NS", "kite": "PAYTM", "type": "EQUITY", "sector": "NEW_AGE_TECH"},
    "ADANIGREEN": {"yfinance": "ADANIGREEN.NS", "kite": "ADANIGREEN", "type": "EQUITY", "sector": "GREEN_ENERGY"},
    "NYKAA": {"yfinance": "NYKAA.NS", "kite": "NYKAA", "type": "EQUITY", "sector": "NEW_AGE_TECH"},
    "POLICYBZR": {"yfinance": "POLICYBZR.NS", "kite": "POLICYBZR", "type": "EQUITY", "sector": "NEW_AGE_TECH"},
    "IREDA": {"yfinance": "IREDA.NS", "kite": "IREDA", "type": "EQUITY", "sector": "GREEN_ENERGY"},
    "HUL": {"yfinance": "HINDUNILVR.NS", "kite": "HINDUNILVR", "type": "FAILSAFE", "sector": "FMCG"},
    "SUNPHARMA": {"yfinance": "SUNPHARMA.NS", "kite": "SUNPHARMA", "type": "FAILSAFE", "sector": "PHARMA"},
    "POWERGRID": {"yfinance": "POWERGRID.NS", "kite": "POWERGRID", "type": "FAILSAFE", "sector": "UTILITIES"},
    "LIQUID_BEES": {"yfinance": "LIQUIDBEES.NS", "kite": "LIQUIDBEES", "type": "FAILSAFE", "sector": "CASH"},
    "GOLD_BEES": {"yfinance": "GOLDBEES.NS", "kite": "GOLDBEES", "type": "FAILSAFE", "sector": "PRECIOUS_METALS"}
}

def initialize_default_map():
    os.makedirs("etc", exist_ok=True)
    with open("etc/ticker_map.json", "w") as f:
        json.dump(DEFAULT_UNIVERSE, f, indent=4)

if __name__ == "__main__":
    initialize_default_map()

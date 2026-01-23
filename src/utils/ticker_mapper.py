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

    def get_yfinance_ticker(self, internal_id):
        return self.mappings.get(internal_id, {}).get("yfinance")

    def get_kite_ticker(self, internal_id):
        return self.mappings.get(internal_id, {}).get("kite")

    def get_all_assets(self):
        return self.mappings

# Default Universe for Initialization
DEFAULT_UNIVERSE = {
    "GOLD": {"yfinance": "GC=F", "kite": "GOLDM", "type": "COMMODITY"},
    "SILVER": {"yfinance": "SI=F", "kite": "SILVERM", "type": "COMMODITY"},
    "CRUDE_OIL": {"yfinance": "CL=F", "kite": "CRUDEOIL", "type": "COMMODITY"},
    "NIFTY_50": {"yfinance": "^NSEI", "kite": "NIFTY 50", "type": "INDEX"},
    "BANK_NIFTY": {"yfinance": "^NSEBANK", "kite": "NIFTY BANK", "type": "INDEX"},
    "RELIANCE": {"yfinance": "RELIANCE.NS", "kite": "RELIANCE", "type": "EQUITY"},
    "HDFCBANK": {"yfinance": "HDFCBANK.NS", "kite": "HDFCBANK", "type": "EQUITY"}
}

def initialize_default_map():
    os.makedirs("etc", exist_ok=True)
    with open("etc/ticker_map.json", "w") as f:
        json.dump(DEFAULT_UNIVERSE, f, indent=4)

if __name__ == "__main__":
    initialize_default_map()

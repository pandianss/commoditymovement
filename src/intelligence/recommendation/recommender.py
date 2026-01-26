import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import PROCESSED_DATA_DIR
from strategies.pnl_engine import PnLEngine
from utils.logger import setup_logger

logger = setup_logger("recommendation_engine", log_file="recommendation.log")

class RecommendationEngine:
    """
    Analyzes multi-asset strategy results and recommends top products
    based on profitability and risk-adjusted returns over various time horizons.
    """
    def __init__(self, performance_file=None):
        self.pnl_engine = PnLEngine()
        self.performance_file = performance_file or os.path.join(PROCESSED_DATA_DIR, "strategy_performance.csv")

    def get_top_recommendations(self, top_n=5, metric="Sharpe Ratio"):
        """
        Loads pre-computed performance and ranks assets.
        """
        if not os.path.exists(self.performance_file):
            logger.error(f"Performance file not found: {self.performance_file}")
            return pd.DataFrame()
            
        df = pd.read_csv(self.performance_file, index_col=0)
        return df.sort_values(by=metric, ascending=False).head(top_n)

    def analyze_period_profitability(self, strategy_returns_dict, periods=[20, 60, 252]):
        """
        Calculates profitability over multiple lookback periods.
        strategy_returns_dict: {AssetID: pd.Series of daily returns}
        periods: List of day counts (e.g. 20 for 1m, 60 for 3m, 252 for 1y)
        """
        recommendations = []
        
        for asset, rets in strategy_returns_dict.items():
            if rets.empty: continue
            
            asset_stats = {"Asset": asset}
            for p in periods:
                p_rets = rets.iloc[-p:] if len(rets) >= p else rets
                if p_rets.empty: continue
                
                cum_ret = (1 + p_rets).cumprod().iloc[-1] - 1
                asset_stats[f"Return_{p}d"] = cum_ret
                
            recommendations.append(asset_stats)
            
        rec_df = pd.DataFrame(recommendations)
        return rec_df

    def spit_recommendations(self, res_df):
        """
        Formats and prints the recommendations to the console.
        """
        print("\n" + "="*50)
        print("          UNIVERSAL RECOMMENDATION ENGINE          ")
        print("="*50)
        
        # Determine best asset for shortest and longest periods
        cols = [c for c in res_df.columns if c.startswith("Return_")]
        if not cols:
            print("No period data available.")
            return

        for col in cols:
            best = res_df.sort_values(by=col, ascending=False).iloc[0]
            print(f"Top Asset ({col.replace('Return_', '')}): {best['Asset']} | Profitability: {best[col]:.2%}")
        
        print("="*50 + "\n")

if __name__ == "__main__":
    # Internal Demo/Test
    recommender = RecommendationEngine()
    # Mock data for demonstration if called directly
    mock_data = pd.DataFrame([
        {"Asset": "GOLD", "Return_20d": 0.05, "Return_252d": 0.25},
        {"Asset": "RELIANCE", "Return_20d": 0.08, "Return_252d": 0.15},
        {"Asset": "NIFTY_50", "Return_20d": 0.02, "Return_252d": 0.18}
    ])
    recommender.spit_recommendations(mock_data)

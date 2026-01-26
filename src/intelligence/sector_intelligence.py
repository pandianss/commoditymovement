import pandas as pd
import numpy as np

class SectorIntelligence:
    """
    Groups assets by sector and identifies coordinated trends,
    emerging sectors, and sector-wide falls.
    """
    def __init__(self, market_df: pd.DataFrame, asset_universe: dict):
        self.df = market_df
        self.universe = asset_universe

    def get_sector_performance(self, lookback_days=22):
        """
        Calculates aggregate performance per sector.
        """
        sector_rets = {}
        
        for asset_id, meta in self.universe.items():
            sector = meta.get("sector", "OTHERS")
            ticker = meta['yfinance']
            ret_col = f"{ticker}_ret_1d"
            
            if ret_col in self.df.columns:
                rets = self.df[ret_col].iloc[-lookback_days:]
                if sector not in sector_rets:
                    sector_rets[sector] = []
                sector_rets[sector].append(rets)
        
        # Aggregate
        sector_summary = []
        for sector, rets_list in sector_rets.items():
            if not rets_list: continue
            
            # Mean daily return across assets in sector
            avg_daily_rets = pd.concat(rets_list, axis=1).mean(axis=1)
            cum_ret = (1 + avg_daily_rets).cumprod().iloc[-1] - 1
            
            # Cohesion: Are all assets moving together? 
            # (Standard deviation of returns across assets at each time step, averaged)
            cohesion = 1 - pd.concat(rets_list, axis=1).std(axis=1).mean()
            
            sector_summary.append({
                "Sector": sector,
                "Cum_Return": cum_ret,
                "Cohesion": cohesion,
                "Asset_Count": len(rets_list)
            })
            
        return pd.DataFrame(sector_summary)

    def identify_emerging_sectors(self, short_p=10, long_p=60):
        """
        Identifies sectors where momentum is accelerating.
        """
        perf_short = self.get_sector_performance(lookback_days=short_p)
        perf_long = self.get_sector_performance(lookback_days=long_p)
        
        if perf_short.empty or perf_long.empty:
            return pd.DataFrame()

        merged = perf_short.merge(perf_long, on="Sector", suffixes=("_short", "_long"))
        # Emerging: Short term performance significantly higher than long term avg
        merged["Acceleration"] = merged["Cum_Return_short"] - (merged["Cum_Return_long"] * (short_p / long_p))
        
        return merged[merged["Acceleration"] > 0.02].sort_values(by="Acceleration", ascending=False)

    def detect_sector_falls(self, threshold=-0.05):
        """
        Detects sectors where the aggregate drawdown is significant.
        """
        perf = self.get_sector_performance(lookback_days=10) # 2-week window for fall detection
        falls = perf[perf["Cum_Return"] < threshold].copy()
        return falls.sort_values(by="Cum_Return")

import pandas as pd
import numpy as np
import os
import sys

class UnicornHunter:
    """
    Algorithmic scanner for identifying 'Unicorn' signatures:
    high acceleration, abnormal relative strength, and structural momentum.
    """
    def __init__(self, market_df: pd.DataFrame, benchmark_ticker="^NSEI"):
        self.df = market_df
        self.benchmark_ticker = benchmark_ticker

    def _calculate_rs_line(self, asset_price, benchmark_price):
        """Calculates Relative Strength line."""
        return asset_price / benchmark_price

    def identify_unicorns(self, lookback_days=126):
        """
        Scans all assets for unicorn signatures.
        """
        unicorns = []
        
        # We need the benchmark (Nifty 50) for RS calculation
        bench_col = f"{self.benchmark_ticker}_Close"
        if bench_col not in self.df.columns:
            return pd.DataFrame()

        bench_price = self.df[bench_col]
        
        # Get all closing price columns
        close_cols = [c for c in self.df.columns if c.endswith("_Close") and c != bench_col]
        
        for col in close_cols:
            asset_id = col.replace("_Close", "")
            asset_price = self.df[col]
            
            # 1. Acceleration: Is the moving average accelerating?
            # 50d MA > 150d MA > 200d MA (Mark Minervini style)
            sma50 = asset_price.rolling(50).mean()
            sma150 = asset_price.rolling(150).mean()
            sma200 = asset_price.rolling(200).mean()
            
            # 2. Relative Strength (RS) Rank
            rs_line = self._calculate_rs_line(asset_price, bench_price)
            # RS momentum: is the RS line hitting new highs?
            rs_new_high = rs_line.iloc[-1] >= rs_line.iloc[-lookback_days:].max()
            
            # 3. Volume Intensity
            vol_col = f"{asset_id}_Volume"
            vol_intensity = 0
            if vol_col in self.df.columns:
                avg_vol = self.df[vol_col].rolling(50).mean()
                vol_intensity = self.df[vol_col].iloc[-1] / avg_vol.iloc[-1] if avg_vol.iloc[-1] > 0 else 0

            # Unicorn Criteria Checklist:
            is_trend_aligned = (asset_price.iloc[-1] > sma50.iloc[-1] > sma150.iloc[-1] > sma200.iloc[-1])
            is_rs_leader = rs_new_high
            is_explosive = vol_intensity > 2.0 # Current volume is 2x average
            
            score = 0
            if is_trend_aligned: score += 40
            if is_rs_leader: score += 40
            if is_explosive: score += 20
            
            if score >= 60:
                unicorns.append({
                    "Asset": asset_id,
                    "Unicorn_Score": score,
                    "RS_Status": "Leader" if is_rs_leader else "Normal",
                    "Trend": "Accelerating" if is_trend_aligned else "Consolidating",
                    "Volume_Intensity": f"{vol_intensity:.1f}x",
                    "Current_Price": asset_price.iloc[-1]
                })

        return pd.DataFrame(unicorns).sort_values(by="Unicorn_Score", ascending=False)

if __name__ == "__main__":
    # Test with dummy data
    dates = pd.date_range("2023-01-01", periods=300)
    data = pd.DataFrame(index=dates)
    data["^NSEI_Close"] = np.linspace(18000, 22000, 300)
    # Unicorn simulation: exponential growth
    data["ZOMATO_Close"] = [100 * (1.01 ** i) for i in range(300)]
    data["ZOMATO_Volume"] = [1000] * 299 + [5000]
    
    hunter = UnicornHunter(data)
    results = hunter.identify_unicorns()
    print("Unicorn Hunt Results:")
    print(results)

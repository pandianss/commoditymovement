import pandas as pd
import numpy as np
import datetime
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestEngine

def create_dummy_data():
    dates = pd.date_range("2024-01-01", periods=100, freq='B')
    
    # Market Data
    price = 100 * (1 + np.random.normal(0, 0.01, size=len(dates))).cumprod()
    market_df = pd.DataFrame({'GOLD': price}, index=dates)
    
    # Allocations (Signal: Alternate Long/Flat every 10 days)
    allocations = []
    for i, date in enumerate(dates):
        if (i // 10) % 2 == 0:
            weight = 0.5 # 50% Long
        else:
            weight = 0.0 # Flat
            
        allocations.append({
            'date': date,
            'asset': 'GOLD',
            'weight': weight
        })
        
    alloc_df = pd.DataFrame(allocations)
    return market_df, alloc_df

def main():
    print("--- Enhanced Backtest Engine Demo ---")
    
    market_df, alloc_df = create_dummy_data()
    
    # Initialize Engine (10bps commission, 5bps slippage)
    engine = BacktestEngine(initial_capital=100_000, commission_bps=10, slippage_bps=5)
    
    print(f"Running Backtest from {market_df.index[0].date()} to {market_df.index[-1].date()}...")
    history = engine.run_backtest(market_df, alloc_df)
    
    print("\n--- Backtest Results ---")
    print(history[['date', 'equity', 'cash']].tail())
    
    # Performance Stats
    start_equity = history.iloc[0]['equity']
    end_equity = history.iloc[-1]['equity']
    ret = (end_equity - start_equity) / start_equity
    
    print(f"\nTotal Return: {ret:.2%}")
    print(f"Total Trades: {len(engine.trades)}")
    
    if len(engine.trades) > 0:
        print("\nLast 5 Trades:")
        for t in engine.trades[-5:]:
            print(f"  {t.date.date()} {t.direction} {t.asset} @ {t.price:.2f} (Cost: {t.cost:.2f})")

if __name__ == "__main__":
    main()

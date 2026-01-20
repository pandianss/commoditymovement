import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, COMMODITIES
from strategies.pnl_engine import PnLEngine
from strategies.persistence_trend import PersistenceTrendStrategy
from strategies.position_sizer import PositionSizer

def main():
    inf_path = os.path.join(PROCESSED_DATA_DIR, "inflection_with_impact.csv")
    store_path = os.path.join(PROCESSED_DATA_DIR, "feature_store_v2.csv")
    
    if not (os.path.exists(inf_path) and os.path.exists(store_path)):
        print("Required data files not found.")
        return
        
    inf_df = pd.read_csv(inf_path, index_col=0, parse_dates=True)
    store_df = pd.read_csv(store_path, index_col=0, parse_dates=True)
    
    engine = PnLEngine()
    sizer = PositionSizer(target_vol=0.15)
    
    # 1. Evaluate Strategy: News Persistence (Event-Driven)
    print("--- Evaluating Strategy: News Persistence Trend ---")
    strat_pt = PersistenceTrendStrategy(min_persistence=60)
    pt_signals_all = strat_pt.generate_signals(store_df, inf_df)
    
    results = []
    
    for commodity in ["GOLD", "CRUDE_OIL"]:
        ticker = COMMODITIES[commodity]
        ret_col = f"{ticker}_ret_1d"
        vol_col = f"{ticker}_vol_20d"
        
        if ret_col not in store_df.columns:
            continue
            
        # Get raw commodity price returns
        price_rets = store_df[ret_col]
        # Get signals for this commodity
        if commodity in pt_signals_all.columns:
            sig = pt_signals_all[commodity]
        else:
            print(f"No signals found for {commodity}")
            sig = pd.Series(0, index=store_df.index)
        
        # Apply Vol Targeting
        vol_weights = store_df[vol_col].apply(lambda x: sizer.calculate_vol_target_weight(x))
        adjusted_signals = sig * vol_weights
        
        # Calculate Returns
        strat_rets = engine.backtest(adjusted_signals, price_rets)
        metrics = engine.calculate_metrics(strat_rets)
        metrics["Commodity"] = commodity
        metrics["Strategy"] = "Persistence Trend"
        results.append(metrics)
        
    res_df = pd.DataFrame(results)
    
    # 2. Evaluate Strategy: Tail-Risk Distribution Play (GOLD only for now)
    print("\n--- Evaluating Strategy: Tail-Risk Distribution Play (GOLD) ---")
    preds_path = os.path.join(PROCESSED_DATA_DIR, "tcn_gold_preds.csv")
    if os.path.exists(preds_path):
        preds_df = pd.read_csv(preds_path, index_col=0, parse_dates=True)
        # Rename columns to floats for the strategy
        preds_df.columns = [float(c) for c in preds_df.columns]
        
        from strategies.distribution_play import DistributionPlayStrategy
        strat_dist = DistributionPlayStrategy(threshold=0.01)
        dist_signals = strat_dist.generate_signals(preds_df)
        
        # Align returns
        ticker_gold = COMMODITIES["GOLD"]
        gold_rets = store_df[f"{ticker_gold}_ret_1d"].loc[dist_signals.index]
        
        # Backtest
        dist_rets = engine.backtest(dist_signals, gold_rets)
        dist_metrics = engine.calculate_metrics(dist_rets)
        dist_metrics["Commodity"] = "GOLD"
        dist_metrics["Strategy"] = "Distribution Play"
        
        # Append to results
        results.append(dist_metrics)
        res_df = pd.DataFrame(results)

    print("\nFinal Strategy Performance Summary:")
    print(res_df[["Commodity", "Strategy", "Sharpe Ratio", "Max Drawdown", "Total Return"]])
    
    # Save results
    output_path = os.path.join(PROCESSED_DATA_DIR, "strategy_performance.csv")
    res_df.to_csv(output_path)
    print(f"\nFull report saved to {output_path}")

if __name__ == "__main__":
    main()

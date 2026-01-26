import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_DIR, ASSET_UNIVERSE
from strategies.pnl_engine import PnLEngine
from strategies.persistence_trend import PersistenceTrendStrategy
from utils.logger import setup_logger
from intelligence.recommendation.recommender import RecommendationEngine
from intelligence.unicorn_hunter import UnicornHunter
from intelligence.macro_intelligence import MacroRegimeClassifier
from intelligence.sector_intelligence import SectorIntelligence

logger = setup_logger("universal_analysis", log_file="universal_analysis.log")

def run_universal_recommendation():
    """
    Main pipeline to run universal backtesting and generate top picks.
    """
    logger.info("Starting Universal Analysis Pipeline...")
    
    store_path = os.path.join(PROCESSED_DATA_DIR, "feature_store_v2.csv")
    if not os.path.exists(store_path):
        logger.error("Feature store not found. Please run data ingestion first.")
        return

    df = pd.read_csv(store_path, index_col=0, parse_dates=True)
    engine = PnLEngine()
    strat = PersistenceTrendStrategy()
    recommender = RecommendationEngine()
    
    # 1. Generate signals for ALL assets
    logger.info("Generating signals for the entire universe...")
    
    # Try persistence signals (news-based)
    inf_path = os.path.join(PROCESSED_DATA_DIR, "inflection_with_impact.csv")
    if os.path.exists(inf_path):
        inf_df = pd.read_csv(inf_path, index_col=0, parse_dates=True)
        news_signals = strat.generate_signals(df, inf_df)
    else:
        news_signals = pd.DataFrame()

    results = {}
    
    # 2. Backtest and collect daily returns
    for asset_id, meta in ASSET_UNIVERSE.items():
        ticker = meta['yfinance']
        ret_col = f"{ticker}_ret_1d"
        
        if ret_col not in df.columns:
            continue
            
        price_rets = df[ret_col]
        
        # Use news signals if available, otherwise fallback to simple momentum (5d vs 20d)
        if asset_id in news_signals.columns and news_signals[asset_id].abs().sum() > 0:
            asset_signals = news_signals[asset_id]
        else:
            # Universal Momentum Baseline: Long if 5d MA > 20d MA
            close_col = f"{ticker}_Close"
            if close_col in df.columns:
                ma5 = df[close_col].rolling(5).mean()
                ma20 = df[close_col].rolling(20).mean()
                asset_signals = (ma5 > ma20).astype(int).replace(0, -1)
            else:
                asset_signals = pd.Series(0, index=df.index)
        
        # Calculate daily strategy returns
        strat_returns = engine.backtest(asset_signals, price_rets)
        results[asset_id] = strat_returns

    # 3. Analyze Period Profitability (Universal coverage!)
    # Periods: 1m, 3m, 6m, 1y
    lookbacks = {"1M": 22, "3M": 66, "6M": 132, "1Y": 252}
    
    period_stats = []
    for asset, rets in results.items():
        if rets.empty: continue
        
        stat = {"Asset": asset, "Type": ASSET_UNIVERSE[asset].get("type")}
        for label, days in lookbacks.items():
            p_rets = rets.iloc[-days:] if len(rets) >= days else rets
            if p_rets.empty: continue
            
            p_metrics = engine.calculate_metrics(p_rets)
            stat[f"Return_{label}"] = p_metrics.get("Total Return", 0)
            stat[f"Sharpe_{label}"] = p_metrics.get("Sharpe Ratio", 0)
            
        period_stats.append(stat)
        
    analysis_df = pd.DataFrame(period_stats)
    
    # 4. Save and Report
    report_path = os.path.join(PROCESSED_DATA_DIR, "universal_recommendations.csv")
    analysis_df.to_csv(report_path, index=False)
    logger.info(f"Universal recommendation report saved to {report_path}")
    
    # 5. Unicorn Hunt
    logger.info("Hunting for High-Potential Unicorns...")
    hunter = UnicornHunter(df)
    unicorns = hunter.identify_unicorns()
    
    # 6. Spit Top Picks
    print("\n" + "#"*60)
    print("         TOP PROFITABLE PRODUCTS BY TIME HORIZON        ")
    print("#"*60)
    
    for label in lookbacks.keys():
        col = f"Return_{label}"
        if col in analysis_df.columns:
            top_pick = analysis_df.sort_values(by=col, ascending=False).iloc[0]
            print(f"[{label}] Best Performer: {top_pick['Asset']} ({top_pick['Type']})")
            print(f"      Profitability: {top_pick[col]:.2%} | Sharpe: {top_pick[f'Sharpe_{label}']:.2f}")
            print("-" * 60)

    if not unicorns.empty:
        print("\n" + "!"*60)
        print("           HIGH POTENTIAL UNICORN ALERT!!!            ")
        print("!"*60)
        for _, u in unicorns.head(3).iterrows():
            print(f"UNICORN: {u['Asset']} | Score: {u['Unicorn_Score']} | Vol: {u['Volume_Intensity']}")
            print(f"      Status: {u['RS_Status']} | Trend: {u['Trend']}")
            print("-" * 60)
    else:
        print("\nNo active Unicorn signatures detected in the current cycle.")

    # 7. Macro Crisis detection
    logger.info("Analyzing Macro Regimes...")
    macro_path = os.path.join(PROCESSED_DATA_DIR, "macro_raw.csv")
    if os.path.exists(macro_path):
        macro_df = pd.read_csv(macro_path, index_col=0, header=[0, 1], parse_dates=True)
        # Flatten multi-index columns for the classifier
        macro_df.columns = [f"{c[1]}_{c[0]}" for c in macro_df.columns]
        
        macro_intel = MacroRegimeClassifier(macro_df)
        regime, dd = macro_intel.detect_crisis()
        failsafe_buckets, stance = macro_intel.get_failsafe_recommendation(regime)
        
        print("\n" + "="*60)
        print(f"           MACRO REGIME: {regime} (DD: {dd:.1%})           ")
        print("="*60)
        print(f"Stance: {stance}")
        if failsafe_buckets:
            print(f"SAFE HARBOR ASSETS: {', '.join(failsafe_buckets)}")
        print("="*60)

    # 8. Sector Intelligence
    logger.info("Analyzing Sector Dynamics...")
    sector_intel = SectorIntelligence(df, ASSET_UNIVERSE)
    emerging = sector_intel.identify_emerging_sectors()
    falls = sector_intel.detect_sector_falls()
    
    print("\n" + "#"*60)
    print("             SECTOR INTELLIGENCE REPORT                ")
    print("#"*60)
    
    if not emerging.empty:
        print(">>> EMERGING SECTORS (Momentum Burst):")
        for _, s in emerging.iterrows():
            print(f"    - {s['Sector']}: Acceleration {s['Acceleration']:.2%} (Cohesion: {s['Cohesion_short']:.2f})")
    
    if not falls.empty:
        print("\n>>> SECTOR-WIDE FALL ALERTS:")
        for _, s in falls.iterrows():
            print(f"    - {s['Sector']}: Sector-wide Drop {s['Cum_Return']:.2%} (All {s['Asset_Count']} components falling)")
    
    if emerging.empty and falls.empty:
        print("No significant sector-wide shifts detected.")
    print("#"*60)
    
    print("\nRecommendation engine identified the above assets as historically strongest.")

if __name__ == "__main__":
    run_universal_recommendation()

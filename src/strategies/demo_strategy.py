import pandas as pd
import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.signal import Signal
from strategies.signal_strategy import ProbabilisticTrendStrategy

def main():
    print("--- Strategy Layer Demo ---")
    
    # 1. Mock signals coming from Intelligence Core
    signals = [
        Signal(
            timestamp_utc=datetime.datetime(2025, 1, 20),
            asset="GOLD",
            signal_type="DIRECTIONAL",
            direction=1.0,
            probability=0.85,
            horizon="5d",
            source="News_Shock_Analyzer"
        ),
        Signal(
            timestamp_utc=datetime.datetime(2025, 1, 20),
            asset="OIL",
            signal_type="DIRECTIONAL",
            direction=-1.0,
            probability=0.40, # Weak signal
            horizon="5d",
            source="TCN_Model"
        ),
        Signal(
            timestamp_utc=datetime.datetime(2025, 1, 21),
            asset="GOLD",
            signal_type="DIRECTIONAL",
            direction=1.0,
            probability=0.92,
            horizon="5d",
            source="TCN_Model"
        )
    ]
    
    print(f"Generated {len(signals)} mock signals.")
    
    # 2. Initialize Strategy (Constraint: High Confidence only)
    strat = ProbabilisticTrendStrategy(confidence_threshold=0.8)
    print("Initialized ProbabilisticTrendStrategy (Threshold=0.8)")
    
    # 3. Generate Allocations
    allocations = strat.generate_allocations(signals)
    
    print("\n--- Portfolio Allocations (Raw) ---")
    if not allocations.empty:
        print(allocations[['date', 'asset', 'weight', 'reason']])
        
        # 4. Apply Risk Budgeting
        capped_alloc = strat.apply_risk_budgeting(allocations.copy(), max_cap=0.20)
        print("\n--- Portfolio Allocations (Risk Capped @ 20%) ---")
        print(capped_alloc[['date', 'asset', 'weight']])
        
        # 5. Check Scenario Exposure
        crash_pnl = strat.determine_exposure(capped_alloc, scenario_shock=-0.10)
        print(f"\nScenario Exposure (10% Crash): {crash_pnl:.2%}")
    else:
        print("No allocations generated.")
        
if __name__ == "__main__":
    main()

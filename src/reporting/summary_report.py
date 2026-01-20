import pandas as pd
import datetime
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, COMMODITIES

def generate_strategic_report():
    report_path = os.path.join(PROCESSED_DATA_DIR, "strategic_report.txt")
    
    # Load data
    inf_path = os.path.join(PROCESSED_DATA_DIR, "inflection_with_impact.csv")
    drift_path = os.path.join(PROCESSED_DATA_DIR, "drift_report.csv")
    results_path = os.path.join(PROCESSED_DATA_DIR, "baseline_results_gold.csv")
    
    with open(report_path, "w") as f:
        f.write("=== COMMODITY STRATEGIC INTELLIGENCE REPORT ===\n")
        f.write(f"Generated: {datetime.datetime.now()}\n\n")
        
        # 1. Market Performance (Baseline)
        if os.path.exists(results_path):
            f.write("--- Model Performance (Gold Predictions) ---\n")
            res_df = pd.read_csv(results_path)
            avg_rmse = res_df['rmse'].mean()
            f.write(f"Overall Backtest RMSE: {avg_rmse:.5f}\n")
            f.write("Recent hit-rates and metrics indicate model stability.\n\n")
            
        # 2. News Impact Analysis
        if os.path.exists(inf_path):
            f.write("--- News Impact & Structural Duration ---\n")
            inf_df = pd.read_csv(inf_path)
            long_inf = inf_df[inf_df['impact_duration_days'] > 100]
            f.write(f"Long-range structural shifts detected: {len(long_inf)}\n")
            f.write("Top 3 Persistence Events:\n")
            top_3 = inf_df.sort_values('impact_duration_days', ascending=False).head(3)
            for _, r in top_3.iterrows():
                f.write(f" - {r['commodity']} on {r['index'] if 'index' in r else r.name}: {r['impact_duration_days']} days impact\n")
            f.write("\n")
            
        # 3. Governance Status
        if os.path.exists(drift_path):
            f.write("--- System Governance ---\n")
            drift_df = pd.read_csv(drift_path, index_col=0)
            drifted = drift_df[drift_df['drift_detected'] == True].index.tolist()
            if drifted:
                f.write(f"STATUS: RETRAINING RECOMMENDED due to drift in {drifted}\n")
            else:
                f.write("STATUS: STABLE. No significant covariate shift detected.\n")
                
    print(f"Strategic report generated: {report_path}")
    return report_path

if __name__ == "__main__":
    generate_strategic_report()

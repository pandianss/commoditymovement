import polars as pl
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("spike_detector", log_file="spike_detector.log")

class SpikeDetector:
    """
    Identifies major price anomalies (spikes/troughs) in asset data.
    Uses Polars for high-performance processing on large datasets.
    """
    
    def __init__(self, spike_threshold_std=2.5, min_pct_change=0.03):
        self.spike_threshold_std = spike_threshold_std
        self.min_pct_change = min_pct_change

    def detect_spikes(self, data_path_or_df, price_col='CLOSE', date_col='DATE'):
        """
        Detects spikes using Polars.
        Args:
            data_path_or_df: Path to Parquet/CSV file OR a Polars DataFrame.
        """
        # Load Data
        if isinstance(data_path_or_df, str):
            if data_path_or_df.endswith('.parquet'):
                df_lazy = pl.scan_parquet(data_path_or_df)
            else:
                df_lazy = pl.scan_csv(data_path_or_df, ignore_errors=True)
        elif isinstance(data_path_or_df, (pl.DataFrame, pl.LazyFrame)):
             df_lazy = data_path_or_df.lazy() if isinstance(data_path_or_df, pl.DataFrame) else data_path_or_df
        else:
            # Fallback for pandas DF (expensive conversion, avoid if possible)
            try:
                import pandas as pd
                if isinstance(data_path_or_df, pd.DataFrame):
                    df_lazy = pl.from_pandas(data_path_or_df).lazy()
            except:
                logger.error(f"Unsupported data type: {type(data_path_or_df)}")
                return None

        # Process Expression
        # 1. Sort by Date
        # 2. Calculate Returns
        # 3. Calculate Rolling Stats (Z-Score)
        
        # Determine grouping
        has_symbol = 'SC_CODE' in df_lazy.collect_schema().names()
        group_col = 'SC_CODE' if has_symbol else None
        
        # Polars Expressions
        # Returns
        ret_expr = pl.col(price_col).pct_change().over(group_col) if group_col else pl.col(price_col).pct_change()
        
        # Statistics (Rolling) - Min periods handled naturally by nulls
        # Note: Polars rolling is strict fast; may need dynamic window workaround if dataset small
        # For Big Data, we assume enough history. 
        # Using rolling_mean/std on the calculated returns
        
        # We need to compute returns first to window over them
        # Polars optimization: use CTE approach
        
        pipeline = (
            df_lazy
            .sort(date_col)
            .with_columns(
                Returns = ret_expr
            )
            .with_columns(
                # Z-Score Calculation
                RollingMean = pl.col('Returns').rolling_mean(window_size=20, min_samples=3).over(group_col) if group_col else pl.col('Returns').rolling_mean(window_size=20, min_samples=3),
                RollingStd = pl.col('Returns').rolling_std(window_size=20, min_samples=3).over(group_col) if group_col else pl.col('Returns').rolling_std(window_size=20, min_samples=3)
            )
            .with_columns(
                Z_Score = (pl.col('Returns') - pl.col('RollingMean')) / pl.col('RollingStd')
            )
            .filter(
                (pl.col('Z_Score').abs() > self.spike_threshold_std) & 
                (pl.col('Returns').abs() > self.min_pct_change)
            )
            .with_columns(
                Type = pl.when(pl.col('Returns') > 0).then(pl.lit('Spike')).otherwise(pl.lit('Trough'))
            )
            .select([date_col, 'SC_CODE', price_col, 'Returns', 'Z_Score', 'Type'] if group_col else [date_col, price_col, 'Returns', 'Z_Score', 'Type'])
        )

        try:
            logger.info("Executing Spike Detection Query (Polars)...")
            result = pipeline.collect()
            logger.info(f"Detected {len(result)} anomalies.")
            return result.to_pandas() # Return as pandas for compatibility with downstream tools for now
        except Exception as e:
            logger.error(f"Error in Polars pipeline: {e}")
            return None

if __name__ == "__main__":
    try:
        # Prefer Parquet
        parquet_path = "data/parquet/bse_history_raw.parquet"
        csv_path = "data/raw/bse_history_raw.csv"
        
        path = parquet_path if os.path.exists(parquet_path) else csv_path
        
        if os.path.exists(path):
            detector = SpikeDetector()
            spikes = detector.detect_spikes(path)
            if spikes is not None:
                print(spikes.head(10))
                spikes.to_csv("data/processed/detected_spikes.csv", index=False)
    except Exception as e:
        print(f"Error: {e}")

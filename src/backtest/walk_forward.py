import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, WALK_FORWARD_CONFIG

class WalkForwardSplitter:
    """
    Splits data into rolling train/test windows.
    """
    def __init__(self, train_window: int, test_window: int, step: int):
        self.train_window = train_window # in days
        self.test_window = test_window # in days
        self.step = step # in days
        
    def split(self, data: pd.DataFrame):
        """
        Generator yielding (train_df, test_df).
        """
        # Ensure data is sorted
        data = data.sort_index()
        dates = data.index
        
        start_date = dates[0]
        end_date = dates[-1]
        
        current_train_start = start_date
        
        while True:
            train_end = current_train_start + datetime.timedelta(days=self.train_window)
            test_start = train_end + datetime.timedelta(hours=1) # Next hour? or Next Day? 
            # If freq is daily, next day. 
            test_end = test_start + datetime.timedelta(days=self.test_window)
            
            if test_end > end_date:
                break
                
            train_mask = (dates >= current_train_start) & (dates <= train_end)
            test_mask = (dates >= test_start) & (dates <= test_end)
            
            train_df = data[train_mask]
            test_df = data[test_mask]
            
            if not train_df.empty and not test_df.empty:
                yield train_df, test_df
                
            current_train_start += datetime.timedelta(days=self.step)


def get_walk_forward_splits(df, config):
    """
    Generates train/test splits for walk-forward validation (Legacy Function).
    """
    splits = []
    start_year = config["start_year"]
    train_size = config["train_size_years"]
    test_size = config["test_size_years"]
    
    # Get total years in df
    df_years = df.index.year.unique().sort_values()
    max_year = df_years.max()
    
    current_test_year = start_year + train_size
    
    while current_test_year <= max_year:
        train_start = datetime.datetime(start_year, 1, 1)
        train_end = datetime.datetime(current_test_year - 1, 12, 31)
        test_start = datetime.datetime(current_test_year, 1, 1)
        test_end = datetime.datetime(current_test_year + test_size - 1, 12, 31)
        
        # Clip to available data
        train_df = df[(df.index >= train_start) & (df.index <= train_end)]
        test_df = df[(df.index >= test_start) & (df.index <= test_end)]
        
        if not test_df.empty:
            splits.append({
                "test_year": current_test_year,
                "train": train_df,
                "test": test_df
            })
            
        current_test_year += test_size
        
    return splits

def run_backtest(model_class, df, target_col, feature_cols, config):
    """
    Runs backtest for a specific model class across all splits.
    """
    splits = get_walk_forward_splits(df, config)
    results = []
    
    all_predictions = []
    all_actuals = []
    
    for split in splits:
        year = split["test_year"]
        X_train = split["train"][feature_cols]
        y_train = split["train"][target_col]
        X_test = split["test"][feature_cols]
        y_test = split["test"][target_col]
        
        # Fit model
        model = model_class()
        model.fit(X_train, y_train)
        
        # Predict
        preds = model.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        
        # Information Coefficient (Rank Correlation) - important for finance
        ic = pd.Series(preds).corr(pd.Series(y_test.values), method='spearman')
        
        print(f"Year {year} | RMSE: {rmse:.5f} | IC: {ic:.4f}")
        
        results.append({
            "year": year,
            "rmse": rmse,
            "mae": mae,
            "ic": ic
        })
        
        all_predictions.extend(preds)
        all_actuals.extend(y_test.values)
        
    overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
    print(f"\nOverall RMSE: {overall_rmse:.5f}")
    
    return results, all_predictions, all_actuals
    
if __name__ == "__main__":
    # Test split logic
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "feature_store.csv"), 
                     index_col=0, parse_dates=True)
    splits = get_walk_forward_splits(df, WALK_FORWARD_CONFIG)
    for s in splits:
        print(f"Year {s['test_year']}: Train until {s['train'].index.max()}, Test {len(s['test'])} days")

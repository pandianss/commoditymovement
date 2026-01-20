import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SequenceGenerator:
    """
    Converts tabular data into sequences for Deep Learning models.
    """
    def __init__(self, window_size=20):
        self.window_size = window_size
        
    def create_sequences(self, df, target_col):
        """
        Creates (N, window_size, dimensions) feature tensors and (N,) target tensors.
        """
        # All columns except target_ as features
        feature_cols = [c for c in df.columns if not c.startswith("target_")]
        
        X_raw = df[feature_cols].values
        y_raw = df[target_col].values
        
        X, y = [], []
        
        for i in range(len(df) - self.window_size):
            X.append(X_raw[i : i + self.window_size])
            y.append(y_raw[i + self.window_size])
            
        return np.array(X), np.array(y)

def prepare_loaders(df, target_col, window_size=20, batch_size=32, train_split=0.8):
    gen = SequenceGenerator(window_size=window_size)
    X, y = gen.create_sequences(df, target_col)
    
    split_idx = int(len(X) * train_split)
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train.shape[2] # return feature dim

if __name__ == "__main__":
    store_path = os.path.join(PROCESSED_DATA_DIR, "feature_store_v2.csv")
    df = pd.read_csv(store_path, index_col=0, parse_dates=True)
    train_loader, val_loader, feat_dim = prepare_loaders(df, "target_GC=F_next_ret")
    
    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch X shape: {X_batch.shape}") # Expect (batch, 20, feat_dim)
    print(f"Batch y shape: {y_batch.shape}")

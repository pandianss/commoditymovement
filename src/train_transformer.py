import torch
import torch.optim as optim
import pandas as pd
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR
from features.sequence_generator import prepare_loaders
from models.transformer_core import TransformerQuantileModel
from models.tcn_engine import quantile_loss

def train_transformer():
    # Load data
    store_path = os.path.join(PROCESSED_DATA_DIR, "feature_store_v2.csv")
    if not os.path.exists(store_path):
        print("Feature store not found.")
        return
        
    df = pd.read_csv(store_path, index_col=0, parse_dates=True)
    
    # Target: Gold
    target_col = "target_GC=F_next_ret"
    quantiles = [0.05, 0.5, 0.95]
    
    train_loader, val_loader, feat_dim = prepare_loaders(df, target_col, window_size=30, batch_size=64)
    
    # Model Config
    model = TransformerQuantileModel(input_dim=feat_dim, d_model=64, nhead=4, 
                                     num_layers=2, quantiles=quantiles)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    epochs = 10
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            preds = model(X)
            loss = quantile_loss(preds, y, quantiles)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.6f}")

    # Save model
    model_path = os.path.join(PROCESSED_DATA_DIR, "transformer_gold_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_transformer()

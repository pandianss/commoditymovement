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
from models.tcn_engine import TCNQuantileModel, quantile_loss

def train_tcn():
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
    # num_channels: [hidden_size] * num_levels
    model = TCNQuantileModel(input_size=feat_dim, num_channels=[32, 32, 32], 
                             quantiles=quantiles, kernel_size=3, dropout=0.2)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
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
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                preds = model(X)
                loss = quantile_loss(preds, y, quantiles)
                val_loss += loss.item()
                
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}")

    # Save model
    model_path = os.path.join(PROCESSED_DATA_DIR, "tcn_gold_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return model

if __name__ == "__main__":
    train_tcn()

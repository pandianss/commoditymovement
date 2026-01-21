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
from core.registry import ModelRegistry, ModelMetadata
from models.experiment import ExperimentLogger
from models.model_selector import ModelSelector
import hashlib

def train_tcn():
    # Initialize experiment tracking
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    experiment = ExperimentLogger(f"tcn_gold_{timestamp}")
    
    # Load data
    store_path = os.path.join(PROCESSED_DATA_DIR, "feature_store.csv")
    if not os.path.exists(store_path):
        print("Feature store not found.")
        return
        
    df = pd.read_csv(store_path, index_col=0, parse_dates=True)
    
    # Calculate feature hash for reproducibility
    feature_cols = [c for c in df.columns if not c.startswith('target_')]
    feature_hash = hashlib.md5('|'.join(sorted(feature_cols)).encode()).hexdigest()[:8]
    
    # Target: Gold
    target_col = "target_GC=F_next_ret"
    quantiles = [0.05, 0.5, 0.95]
    
    # Hyperparameters
    hyperparams = {
        "window_size": 30,
        "batch_size": 64,
        "num_channels": [32, 32, 32],
        "kernel_size": 3,
        "dropout": 0.2,
        "lr": 0.001,
        "epochs": 20
    }
    experiment.log_params(hyperparams)
    
    train_loader, val_loader, feat_dim = prepare_loaders(df, target_col, 
                                                          window_size=hyperparams["window_size"], 
                                                          batch_size=hyperparams["batch_size"])
    
    # Model Config
    model = TCNQuantileModel(input_size=feat_dim, 
                            num_channels=hyperparams["num_channels"], 
                            quantiles=quantiles, 
                            kernel_size=hyperparams["kernel_size"], 
                            dropout=hyperparams["dropout"])
    
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])
    
    epochs = hyperparams["epochs"]
    print(f"Starting training for {epochs} epochs...")
    
    best_val_loss = float('inf')
    
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
        
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
                
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # Log final metrics
    final_metrics = {
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "best_val_loss": best_val_loss
    }
    experiment.log_metrics(final_metrics)
    
    # Save model
    model_id = f"tcn_gold_{timestamp}"
    model_path = os.path.join(PROCESSED_DATA_DIR, f"{model_id}.pth")
    torch.save(model.state_dict(), model_path)
    experiment.log_artifact(model_path, "model")
    print(f"Model saved to {model_path}")
    
    # Register in Model Registry
    registry = ModelRegistry()
    metadata = ModelMetadata(
        model_id=model_id,
        version="1.0",
        model_path=model_path,
        feature_set_version="v1",
        feature_hash=feature_hash,
        training_window={
            "start": str(df.index.min().date()),
            "end": str(df.index.max().date())
        },
        hyperparameters=hyperparams,
        metrics=final_metrics,
        status="candidate"
    )
    
    registry.register(metadata)
    print(f"Model registered in registry: {model_id}")
    
    # Auto-promote if better than current champion
    selector = ModelSelector(registry)
    promoted = selector.auto_promote_if_better(model_id, primary_metric="val_loss")
    
    if promoted:
        print(f"✓ Model promoted to CHAMPION")
    else:
        print(f"Model registered as CANDIDATE (did not outperform champion)")
    
    experiment.finalize()
    return model

if __name__ == "__main__":
    train_tcn()

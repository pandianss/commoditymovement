import torch
import torch.optim as optim
import pandas as pd
import os
import sys
import numpy as np
import argparse

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, ASSET_UNIVERSE
from features.sequence_generator import prepare_loaders
from models.tcn_engine import TCNQuantileModel, quantile_loss
from core.registry import ModelRegistry, ModelMetadata
from models.experiment import ExperimentLogger
from models.model_selector import ModelSelector
from utils.logger import setup_logger
import hashlib

logger = setup_logger("sector_training", log_file="sector_training.log")

def get_sector_constituents(sector):
    """Get all assets belonging to a sector."""
    return [k for k, v in ASSET_UNIVERSE.items() if v.get('sector') == sector]

def aggregate_sector_data(df, sector):
    """
    Aggregate sector data by averaging constituent returns.
    This creates a synthetic 'sector index' for training.
    """
    constituents = get_sector_constituents(sector)
    logger.info(f"Sector {sector} has {len(constituents)} constituents")
    
    # Find return columns for constituents
    sector_cols = []
    for asset in constituents:
        yf_ticker = ASSET_UNIVERSE[asset]['yfinance']
        ret_col = f"target_{yf_ticker}_next_ret"
        if ret_col in df.columns:
            sector_cols.append(ret_col)
    
    if not sector_cols:
        logger.warning(f"No return columns found for sector {sector}")
        return None
    
    # Average returns across constituents
    sector_return = df[sector_cols].mean(axis=1)
    df_sector = df.copy()
    df_sector[f'target_{sector}_sector_ret'] = sector_return
    
    logger.info(f"Created sector index with {len(sector_cols)} constituents")
    return df_sector, f'target_{sector}_sector_ret'

def train_sector_model(sector, epochs=20, window_size=30):
    """
    Train a TCN model for a specific sector.
    
    Args:
        sector: Sector name (e.g., 'PRECIOUS_METALS', 'ENERGY', 'BFSI')
        epochs: Number of training epochs
        window_size: Sequence window size
    """
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    experiment = ExperimentLogger(f"tcn_{sector}_{timestamp}")
    
    # Load feature store
    store_path = os.path.join(PROCESSED_DATA_DIR, "feature_store.csv")
    if not os.path.exists(store_path):
        logger.error("Feature store not found. Run feature engineering first.")
        return None
    
    df = pd.read_csv(store_path, index_col=0, parse_dates=True)
    
    # Aggregate sector data
    df_sector, target_col = aggregate_sector_data(df, sector)
    if df_sector is None:
        logger.error(f"Failed to aggregate data for sector {sector}")
        return None
    
    # Calculate feature hash
    feature_cols = [c for c in df_sector.columns if not c.startswith('target_')]
    feature_hash = hashlib.md5('|'.join(sorted(feature_cols)).encode()).hexdigest()[:8]
    
    # Hyperparameters
    hyperparams = {
        "sector": sector,
        "window_size": window_size,
        "batch_size": 64,
        "num_channels": [32, 32, 32],
        "kernel_size": 3,
        "dropout": 0.2,
        "lr": 0.001,
        "epochs": epochs
    }
    experiment.log_params(hyperparams)
    
    # Prepare data loaders
    quantiles = [0.05, 0.5, 0.95]
    train_loader, val_loader, feat_dim = prepare_loaders(
        df_sector, target_col, 
        window_size=window_size, 
        batch_size=hyperparams["batch_size"]
    )
    
    # Initialize model
    model = TCNQuantileModel(
        input_size=feat_dim, 
        num_channels=hyperparams["num_channels"], 
        quantiles=quantiles, 
        kernel_size=hyperparams["kernel_size"], 
        dropout=hyperparams["dropout"]
    )
    
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])
    
    logger.info(f"Starting training for {sector} - {epochs} epochs...")
    
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
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
    
    # Log final metrics
    final_metrics = {
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "best_val_loss": best_val_loss
    }
    experiment.log_metrics(final_metrics)
    
    # Save model
    model_id = f"tcn_{sector}_{timestamp}"
    model_path = os.path.join(PROCESSED_DATA_DIR, f"{model_id}.pth")
    torch.save(model.state_dict(), model_path)
    experiment.log_artifact(model_path, "model")
    logger.info(f"Model saved to {model_path}")
    
    # Register in Model Registry
    registry = ModelRegistry()
    metadata = ModelMetadata(
        model_id=model_id,
        version="1.0",
        model_path=model_path,
        feature_set_version="v1",
        feature_hash=feature_hash,
        training_window={
            "start": str(df_sector.index.min().date()),
            "end": str(df_sector.index.max().date())
        },
        hyperparameters=hyperparams,
        metrics=final_metrics,
        status="candidate"
    )
    
    registry.register(metadata)
    logger.info(f"Model registered: {model_id}")
    
    # Auto-promote if better than current champion
    selector = ModelSelector(registry)
    promoted = selector.auto_promote_if_better(model_id, primary_metric="val_loss")
    
    if promoted:
        logger.info(f"✓ Model promoted to CHAMPION for {sector}")
    else:
        logger.info(f"Model registered as CANDIDATE for {sector}")
    
    experiment.finalize()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TCN models for sectors')
    parser.add_argument('--sector', type=str, default='PRECIOUS_METALS',
                        help='Sector to train (e.g., PRECIOUS_METALS, ENERGY, BFSI)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--all', action='store_true',
                        help='Train models for all high-potential sectors')
    
    args = parser.parse_args()
    
    if args.all:
        # Train for top sectors identified from analysis
        top_sectors = ['PRECIOUS_METALS', 'ENERGY', 'BFSI', 'IT']
        for sector in top_sectors:
            logger.info(f"\n{'='*60}\nTraining model for {sector}\n{'='*60}")
            train_sector_model(sector, epochs=args.epochs)
    else:
        train_sector_model(args.sector, epochs=args.epochs)

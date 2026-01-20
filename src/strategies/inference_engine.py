import torch
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR
from features.sequence_generator import SequenceGenerator
from models.tcn_engine import TCNQuantileModel

def run_tcn_inference(df, target_col, model_path, window_size=30):
    # Prepare data
    gen = SequenceGenerator(window_size=window_size)
    X, y = gen.create_sequences(df, target_col)
    X_tensor = torch.FloatTensor(X)
    
    # Feature dim
    feat_dim = X.shape[2]
    quantiles = [0.05, 0.5, 0.95]
    
    # Load Model
    model = TCNQuantileModel(input_size=feat_dim, num_channels=[32, 32, 32], 
                             quantiles=quantiles, kernel_size=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        preds_dict = model(X_tensor)
        
    # Convert to DataFrame
    preds_df = pd.DataFrame(index=df.index[window_size:])
    for q in quantiles:
        preds_df[q] = preds_dict[q].numpy().flatten()
        
    return preds_df

if __name__ == "__main__":
    store_path = os.path.join(PROCESSED_DATA_DIR, "feature_store_v2.csv")
    df = pd.read_csv(store_path, index_col=0, parse_dates=True)
    
    model_path = os.path.join(PROCESSED_DATA_DIR, "tcn_gold_model.pth")
    if os.path.exists(model_path):
        print("Running TCN Inference for GOLD...")
        preds = run_tcn_inference(df, "target_GC=F_next_ret", model_path)
        output_path = os.path.join(PROCESSED_DATA_DIR, "tcn_gold_preds.csv")
        preds.to_csv(output_path)
        print(f"Predictions saved to {output_path}")
    else:
        print("Model file not found.")

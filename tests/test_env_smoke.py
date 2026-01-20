import sys
import os

# Add src to path so we can import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    from models.tcn_engine import TCNQuantileModel
    from models.transformer_core import TransformerQuantileModel

    # Smoke Test TCN
    print("Initializing TCNQuantileModel...")
    tcn = TCNQuantileModel(input_size=10, num_channels=[16, 32])
    print("TCN Initialization Successful.")

    # Smoke Test Transformer
    print("Initializing TransformerQuantileModel...")
    transformer = TransformerQuantileModel(input_dim=10)
    print("Transformer Initialization Successful.")

    print("\nEnvironment smoke test passed!")

except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)

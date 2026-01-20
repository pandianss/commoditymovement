import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerQuantileModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, 
                 dropout=0.1, quantiles=[0.05, 0.5, 0.95]):
        super(TransformerQuantileModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.d_model = d_model
        
        # Quantile heads
        self.heads = nn.ModuleDict({
            f"q_{str(q).replace('.', '_')}": nn.Linear(d_model, 1) for q in quantiles
        })
        self.quantiles = quantiles

    def forward(self, x):
        # x: (N, L, Cin)
        x = self.input_proj(x) # (N, L, d_model)
        x = x.transpose(0, 1) # (L, N, d_model)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x) # (L, N, d_model)
        
        # We take the mean or last step? Let's take the last step for time-series forecasting
        last_step = output[-1, :, :] # (N, d_model)
        
        preds = {q: self.heads[f"q_{str(q).replace('.', '_')}"](last_step) for q in self.quantiles}
        return preds

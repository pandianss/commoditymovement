import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class ChausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(ChausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=self.padding, dilation=dilation))
        
    def forward(self, x):
        # x: (N, C, L)
        x = self.conv(x)
        return x[:, :, :-self.padding] # Trim the future padding

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = ChausalConv1d(n_inputs, n_outputs, kernel_size, stride=stride, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = ChausalConv1d(n_outputs, n_outputs, kernel_size, stride=stride, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        num_channels: list of output channels for each block
        """
        super(TemporalConvolutionalNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x expected shape: (N, L, C) -> needs to be (N, C, L) for Conv1d
        x = x.transpose(1, 2)
        return self.network(x).transpose(1, 2)

class TCNQuantileModel(nn.Module):
    def __init__(self, input_size, num_channels, quantiles=[0.05, 0.5, 0.95], kernel_size=3, dropout=0.2):
        super(TCNQuantileModel, self).__init__()
        self.tcn = TemporalConvolutionalNetwork(input_size, num_channels, kernel_size, dropout)
        # Quantile heads - rename to avoid dots in keys
        self.heads = nn.ModuleDict({
            f"q_{str(q).replace('.', '_')}": nn.Linear(num_channels[-1], 1) for q in quantiles
        })
        self.quantiles = quantiles
        
    def forward(self, x):
        # x: (N, L, C)
        output = self.tcn(x) # (N, L, out_C)
        last_step = output[:, -1, :] # We only care about the last time step for prediction
        
        preds = {q: self.heads[f"q_{str(q).replace('.', '_')}"](last_step) for q in self.quantiles}
        return preds

def quantile_loss(preds, target, quantiles):
    losses = []
    for q in quantiles:
        pred = preds[q].squeeze()
        errors = target - pred
        loss = torch.max((q - 1) * errors, q * errors)
        losses.append(loss.mean())
    return sum(losses)

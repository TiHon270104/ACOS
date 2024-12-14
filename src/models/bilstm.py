import torch
import torch.nn as nn

class BiLSTMWithResidual(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # BiLSTM
        lstm_out, _ = self.lstm(x)
        
        # Residual connection
        output = self.layer_norm(x + self.dropout(lstm_out))
        
        return output
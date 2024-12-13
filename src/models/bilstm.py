import torch.nn as nn
from modules.attention import MultiHeadAttention

class BiLSTMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=config['hidden_size'],
            hidden_size=config['hidden_size'] // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        self.attention = MultiHeadAttention(
            config['hidden_size'],
            num_heads=8
        )
        
    def forward(self, x, mask=None):
        # BiLSTM encoding
        lstm_out, _ = self.lstm(x)
        
        # Self attention
        if mask is not None:
            attn_out = self.attention(lstm_out, lstm_out, lstm_out, mask)
        else:
            attn_out = self.attention(lstm_out, lstm_out, lstm_out)
            
        return attn_out
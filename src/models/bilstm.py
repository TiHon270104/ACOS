import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=config['hidden_size'],
            hidden_size=config['hidden_size'] // 2,
            num_layers=config.get('lstm_layers', 1),
            bidirectional=True,
            batch_first=True,
            dropout=config.get('dropout', 0.1) if config.get('lstm_layers', 1) > 1 else 0
        )
        
    def forward(self, x, mask=None):
        # Pack padded sequence if mask is provided
        if mask is not None:
            lengths = mask.sum(dim=1)
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed_x)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            output, _ = self.lstm(x)
            
        return output
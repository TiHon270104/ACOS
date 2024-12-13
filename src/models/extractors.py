import torch.nn as nn

class AspectExtractor(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        return self.extractor(x).squeeze(-1)

class OpinionExtractor(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        return self.extractor(x).squeeze(-1)
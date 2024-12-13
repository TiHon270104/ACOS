import torch.nn as nn

class AspectExtractor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.extractor(x).squeeze(-1)

class OpinionExtractor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.extractor(x).squeeze(-1)
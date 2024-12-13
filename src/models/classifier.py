import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)
        )
        
    def forward(self, x):
        return self.classifier(x)

class CategoryClassifier(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 5)
        )
        
    def forward(self, x):
        return self.classifier(x)
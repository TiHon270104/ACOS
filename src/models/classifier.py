import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 3)
        
    def forward(self, x):
        return self.classifier(x)

class CategoryClassifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 5)
        
    def forward(self, x):
        return self.classifier(x)
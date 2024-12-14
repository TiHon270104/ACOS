import torch
import torch.nn as nn

class AspectOpinionExtractor(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        
        # Aspect extraction
        self.aspect_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)  # Binary classification
        )
        
        # Opinion extraction  
        self.opinion_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)  # Binary classification
        )
        
    def forward(self, x):
        aspect_logits = self.aspect_extractor(x)
        opinion_logits = self.opinion_extractor(x)
        
        return aspect_logits, opinion_logits
import torch
import torch.nn as nn

class SentimentCategoryClassifier(nn.Module):
    def __init__(self, hidden_size, num_sentiments=3, num_categories=5):
        super().__init__()
        
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_sentiments)
        )
        
        self.category_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_categories)
        )
        
    def forward(self, x):
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        sentiment_logits = self.sentiment_classifier(x)
        category_logits = self.category_classifier(x)
        
        return sentiment_logits, category_logits
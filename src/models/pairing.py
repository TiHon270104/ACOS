import torch
import torch.nn as nn

class AspectOpinionPairing(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.aspect_proj = nn.Linear(hidden_size, hidden_size)
        self.opinion_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, aspect_repr, opinion_repr):
        # Project representations
        aspect_proj = self.aspect_proj(aspect_repr)
        opinion_proj = self.opinion_proj(opinion_repr)
        
        # Efficient matrix multiplication
        scores = torch.bmm(aspect_proj, opinion_proj.transpose(1, 2))
        
        return scores
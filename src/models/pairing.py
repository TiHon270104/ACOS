import torch
import torch.nn as nn

class AspectOpinionPairing(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.biaffine = nn.Bilinear(hidden_size, hidden_size, 1)
        
    def forward(self, aspect_repr, opinion_repr):
        scores = self.biaffine(aspect_repr, opinion_repr)
        return scores.squeeze(-1)
import torch.nn as nn
from transformers import AutoModel

class PhoBertEncoder(nn.Module):
    def __init__(self, model_name='vinai/phobert-base', dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        return sequence_output
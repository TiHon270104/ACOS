import torch
import torch.nn as nn
from models.bert_encoder import PhoBertEncoder
from models.bilstm import BiLSTMWithResidual
from modules.attention import MultiHeadAttention
from models.extractors import AspectOpinionExtractor
from models.pairing import AspectOpinionPairing
from models.classifier import SentimentCategoryClassifier

class ACOSModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # PhoBERT encoder
        self.encoder = PhoBertEncoder(config['model_name'])
        
        # Custom self-attention
        self.self_attention = MultiHeadAttention(
            config['hidden_size'],
            config['num_attention_heads']
        )
        
        # BiLSTM
        self.bilstm = BiLSTMWithResidual(
            config['hidden_size'],
            config['hidden_size']
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(config['hidden_size'] * 2, config['hidden_size']),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Aspect-Opinion extraction
        self.extractors = AspectOpinionExtractor(config['hidden_size'])
        
        # Aspect-Opinion pairing
        self.pairing = AspectOpinionPairing(config['hidden_size'])
        
        # Sentiment & Category classification
        self.classifier = SentimentCategoryClassifier(
            config['hidden_size'],
            config['num_sentiments'],
            config['num_categories']
        )
        
    def forward(self, input_ids, attention_mask):
        # Encode text
        sequence_output = self.encoder(input_ids, attention_mask)
        
        # Self attention
        attended = self.self_attention(sequence_output, attention_mask)
        
        # BiLSTM
        lstm_output = self.bilstm(attended)
        
        # Feature fusion
        fused = self.feature_fusion(
            torch.cat([attended, lstm_output], dim=-1)
        )
        
        # Extract aspects & opinions
        aspect_logits, opinion_logits = self.extractors(fused)
        
        # Aspect-Opinion pairing
        pairing_matrix = self.pairing(
            fused,
            aspect_logits.softmax(dim=-1),
            opinion_logits.softmax(dim=-1)
        )
        
        # Classification
        sentiment_logits, category_logits = self.classifier(fused)
        
        return {
            'aspect_logits': aspect_logits,
            'opinion_logits': opinion_logits,
            'pairing_matrix': pairing_matrix,
            'sentiment_logits': sentiment_logits,
            'category_logits': category_logits
        }
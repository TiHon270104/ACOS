import torch.nn as nn
from .bert_encoder import PhoBERTEncoder
from .bilstm import BiLSTMEncoder
from .extractors import AspectExtractor, OpinionExtractor
from .pairing import AspectOpinionPairing
from .classifier import SentimentClassifier, CategoryClassifier

class ABSAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # PhoBERT encoder
        self.encoder = PhoBERTEncoder(config['model_name'])
        
        # Freeze embeddings if specified
        if config.get('freeze_embeddings', False):
            for param in self.encoder.encoder.embeddings.parameters():
                param.requires_grad = False
        
        # BiLSTM encoder
        self.bilstm = BiLSTMEncoder(config)
        
        hidden_size = config['hidden_size']
        dropout = config.get('dropout', 0.1)
        
        # Task-specific layers with dropout
        self.dropout = nn.Dropout(dropout)
        
        self.aspect_extractor = AspectExtractor(hidden_size, dropout)
        self.opinion_extractor = OpinionExtractor(hidden_size, dropout)
        self.pairing = AspectOpinionPairing(hidden_size)
        
        self.sentiment_classifier = SentimentClassifier(hidden_size, dropout)
        self.category_classifier = CategoryClassifier(hidden_size, dropout)
        
    def forward(self, input_ids, attention_mask):
        # Encode text
        encoder_output = self.encoder(input_ids, attention_mask)
        
        # BiLSTM + Attention
        sequence_output = self.bilstm(encoder_output, attention_mask)
        sequence_output = self.dropout(sequence_output)
        
        # Extract aspects and opinions
        aspect_logits = self.aspect_extractor(sequence_output)
        opinion_logits = self.opinion_extractor(sequence_output)
        
        # Aspect-Opinion pairing
        pairing_scores = self.pairing(sequence_output, sequence_output)
        
        # Classification
        pooled = sequence_output.mean(dim=1)
        sentiment_logits = self.sentiment_classifier(pooled)
        category_logits = self.category_classifier(pooled)
        
        return {
            'aspect': aspect_logits,
            'opinion': opinion_logits,
            'pairing': pairing_scores,
            'sentiment': sentiment_logits,
            'category': category_logits
        }
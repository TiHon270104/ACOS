import torch
import torch.nn as nn

class ABSALoss:
    def __init__(self):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def __call__(self, predictions, labels):
        total_loss = 0
        losses = {}
        
        # Aspect extraction loss
        if 'aspect' in predictions and 'aspect_mask' in labels:
            aspect_loss = self.bce_loss(predictions['aspect'], labels['aspect_mask'])
            total_loss += aspect_loss
            losses['aspect_loss'] = aspect_loss.item()
        
        # Opinion extraction loss
        if 'opinion' in predictions and 'opinion_mask' in labels:
            opinion_loss = self.bce_loss(predictions['opinion'], labels['opinion_mask'])
            total_loss += opinion_loss
            losses['opinion_loss'] = opinion_loss.item()
        
        # Pairing loss
        if 'pairing' in predictions and 'pairing_mask' in labels:
            pairing_loss = self.bce_loss(predictions['pairing'], labels['pairing_mask'])
            total_loss += pairing_loss
            losses['pairing_loss'] = pairing_loss.item()
        
        # Sentiment classification loss
        if 'sentiment' in predictions and 'sentiment' in labels:
            sentiment_loss = self.ce_loss(predictions['sentiment'], labels['sentiment'])
            total_loss += sentiment_loss
            losses['sentiment_loss'] = sentiment_loss.item()
        
        # Category classification loss
        if 'category' in predictions and 'category' in labels:
            category_loss = self.ce_loss(predictions['category'], labels['category'])
            total_loss += category_loss
            losses['category_loss'] = category_loss.item()
        
        losses['total_loss'] = total_loss.item()
        return total_loss, losses
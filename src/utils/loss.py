import torch
import torch.nn as nn
import torch.nn.functional as F

class ABSALoss:
    def __init__(self):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def __call__(self, predictions, labels):
        total_loss = 0
        losses = {}
        
        # Aspect extraction loss
        if 'aspect' in predictions and 'aspect_mask' in labels:
            aspect_loss = self.bce_loss(
                predictions['aspect'],  # [batch_size, seq_len]
                labels['aspect_mask'].float()  # [batch_size, seq_len]
            )
            total_loss += aspect_loss
            losses['aspect_loss'] = aspect_loss.item()
        
        # Opinion extraction loss
        if 'opinion' in predictions and 'opinion_mask' in labels:
            opinion_loss = self.bce_loss(
                predictions['opinion'],  # [batch_size, seq_len]
                labels['opinion_mask'].float()  # [batch_size, seq_len]
            )
            total_loss += opinion_loss
            losses['opinion_loss'] = opinion_loss.item()
        
        # Pairing loss
        if 'pairing' in predictions and 'pairing_mask' in labels:
            # Reshape predictions if needed
            if predictions['pairing'].dim() == 2:
                batch_size = predictions['pairing'].size(0)
                seq_len = int(torch.sqrt(predictions['pairing'].size(1)))
                predictions_reshaped = predictions['pairing'].view(batch_size, seq_len, seq_len)
            else:
                predictions_reshaped = predictions['pairing']
                
            pairing_loss = self.bce_loss(
                predictions_reshaped,  # [batch_size, seq_len, seq_len]
                labels['pairing_mask'].float()  # [batch_size, seq_len, seq_len]
            )
            total_loss += pairing_loss
            losses['pairing_loss'] = pairing_loss.item()
        
        # Sentiment classification loss
        if 'sentiment' in predictions and 'sentiment' in labels:
            sentiment_loss = self.ce_loss(
                predictions['sentiment'],  # [batch_size, num_classes]
                labels['sentiment']  # [batch_size]
            )
            total_loss += sentiment_loss
            losses['sentiment_loss'] = sentiment_loss.item()
        
        # Category classification loss
        if 'category' in predictions and 'category' in labels:
            category_loss = self.ce_loss(
                predictions['category'],  # [batch_size, num_classes]
                labels['category']  # [batch_size]
            )
            total_loss += category_loss
            losses['category_loss'] = category_loss.item()
        
        losses['total_loss'] = total_loss.item()
        return total_loss, losses
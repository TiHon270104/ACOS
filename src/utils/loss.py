import torch
import torch.nn as nn
import torch.nn.functional as F

class ACOSLoss:
    def __init__(self, config):
        self.aspect_criterion = nn.CrossEntropyLoss()
        self.opinion_criterion = nn.CrossEntropyLoss()
        self.sentiment_criterion = nn.CrossEntropyLoss()
        self.category_criterion = nn.CrossEntropyLoss()
        self.pairing_criterion = nn.BCEWithLogitsLoss()
        
        # Loss weights
        self.w1 = config['aspect_loss_weight']
        self.w2 = config['opinion_loss_weight']
        self.w3 = config['pairing_loss_weight']
        self.w4 = config['sentiment_loss_weight']
        self.w5 = config['category_loss_weight']
        
    def compute_loss(self, outputs, targets):
        # Aspect extraction loss
        aspect_loss = self.aspect_criterion(
            outputs['aspect_logits'].view(-1, 2),
            targets['aspect_labels'].view(-1)
        )
        
        # Opinion extraction loss
        opinion_loss = self.opinion_criterion(
            outputs['opinion_logits'].view(-1, 2),
            targets['opinion_labels'].view(-1)
        )
        
        # Pairing loss
        pairing_loss = self.pairing_criterion(
            outputs['pairing_matrix'],
            targets['pairing_labels']
        )
        
        # Sentiment classification loss
        sentiment_loss = self.sentiment_criterion(
            outputs['sentiment_logits'],
            targets['sentiment_labels']
        )
        
        # Category classification loss
        category_loss = self.category_criterion(
            outputs['category_logits'],
            targets['category_labels']
        )
        
        # Total loss
        total_loss = (
            self.w1 * aspect_loss +
            self.w2 * opinion_loss +
            self.w3 * pairing_loss +
            self.w4 * sentiment_loss +
            self.w5 * category_loss
        )
        
        return {
            'total_loss': total_loss,
            'aspect_loss': aspect_loss,
            'opinion_loss': opinion_loss,
            'pairing_loss': pairing_loss,
            'sentiment_loss': sentiment_loss,
            'category_loss': category_loss
        }
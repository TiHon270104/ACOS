import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

class ACOSMetrics:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.aspect_preds = []
        self.aspect_labels = []
        self.opinion_preds = []
        self.opinion_labels = []
        self.sentiment_preds = []
        self.sentiment_labels = []
        self.category_preds = []
        self.category_labels = []
        self.pairing_preds = []
        self.pairing_labels = []
        
    def update(self, outputs, targets):
        # Aspect
        aspect_pred = outputs['aspect_logits'].argmax(dim=-1)
        self.aspect_preds.extend(aspect_pred.cpu().numpy())
        self.aspect_labels.extend(targets['aspect_labels'].cpu().numpy())
        
        # Opinion
        opinion_pred = outputs['opinion_logits'].argmax(dim=-1)
        self.opinion_preds.extend(opinion_pred.cpu().numpy())
        self.opinion_labels.extend(targets['opinion_labels'].cpu().numpy())
        
        # Sentiment
        sentiment_pred = outputs['sentiment_logits'].argmax(dim=-1)
        self.sentiment_preds.extend(sentiment_pred.cpu().numpy())
        self.sentiment_labels.extend(targets['sentiment_labels'].cpu().numpy())
        
        # Category
        category_pred = outputs['category_logits'].argmax(dim=-1)
        self.category_preds.extend(category_pred.cpu().numpy())
        self.category_labels.extend(targets['category_labels'].cpu().numpy())
        
        # Pairing
        pairing_pred = (outputs['pairing_matrix'] > 0).float()
        self.pairing_preds.extend(pairing_pred.cpu().numpy())
        self.pairing_labels.extend(targets['pairing_labels'].cpu().numpy())
        
    def compute(self):
        metrics = {}
        
        # Aspect metrics
        metrics['aspect_f1'] = f1_score(
            self.aspect_labels,
            self.aspect_preds,
            average='macro'
        )
        
        # Opinion metrics
        metrics['opinion_f1'] = f1_score(
            self.opinion_labels,
            self.opinion_preds,
            average='macro'
        )
        
        # Sentiment metrics
        metrics['sentiment_f1'] = f1_score(
            self.sentiment_labels,
            self.sentiment_preds,
            average='macro'
        )
        
        # Category metrics
        metrics['category_f1'] = f1_score(
            self.category_labels,
            self.category_preds,
            average='macro'
        )
        
        # Pairing metrics
        metrics['pairing_f1'] = f1_score(
            self.pairing_labels,
            self.pairing_preds,
            average='binary'
        )
        
        return metrics
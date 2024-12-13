import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Metrics:
    def __init__(self):
        self.sentiment_labels = ['positive', 'negative', 'neutral']
        self.category_labels = ['APPEARANCE', 'TECHNICAL', 'SPECIALIZE', 'OTHER', 'CHARACTERISTIC']

    def compute_metrics(self, predictions, labels):
        """Tính toán tất cả metrics"""
        results = {}
        
        # Accuracy metrics
        results.update(self.compute_accuracy(predictions, labels))
        
        # F1 scores
        results.update(self.compute_f1(predictions, labels))
        
        return results

    def compute_accuracy(self, predictions, labels):
        results = {}
        
        # Sentiment accuracy
        if 'sentiment' in predictions and 'sentiment' in labels:
            sent_preds = torch.argmax(predictions['sentiment'], dim=1)
            sent_acc = accuracy_score(labels['sentiment'].cpu(), sent_preds.cpu())
            results['sentiment_accuracy'] = sent_acc
            
        # Category accuracy
        if 'category' in predictions and 'category' in labels:
            cat_preds = torch.argmax(predictions['category'], dim=1)
            cat_acc = accuracy_score(labels['category'].cpu(), cat_preds.cpu())
            results['category_accuracy'] = cat_acc

        return results

    def compute_f1(self, predictions, labels):
        results = {}
        
        # Aspect F1
        if 'aspect' in predictions and 'aspect_mask' in labels:
            aspect_preds = (predictions['aspect'] > 0).float()
            aspect_f1 = f1_score(
                labels['aspect_mask'].cpu().numpy().flatten(),
                aspect_preds.cpu().numpy().flatten(),
                average='binary'
            )
            results['aspect_f1'] = aspect_f1
            
        # Opinion F1
        if 'opinion' in predictions and 'opinion_mask' in labels:
            opinion_preds = (predictions['opinion'] > 0).float()
            opinion_f1 = f1_score(
                labels['opinion_mask'].cpu().numpy().flatten(),
                opinion_preds.cpu().numpy().flatten(),
                average='binary'
            )
            results['opinion_f1'] = opinion_f1

        return results

    def plot_confusion_matrices(self, predictions, labels, save_path):
        """Vẽ và lưu confusion matrices"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # Sentiment confusion matrix
        if 'sentiment' in predictions and 'sentiment' in labels:
            sent_preds = torch.argmax(predictions['sentiment'], dim=1).cpu()
            sent_true = labels['sentiment'].cpu()
            
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(sent_true, sent_preds)
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.sentiment_labels, yticklabels=self.sentiment_labels)
            plt.title('Sentiment Confusion Matrix')
            plt.savefig(f'{save_path}/sentiment_cm.png')
            plt.close()
            
        # Category confusion matrix
        if 'category' in predictions and 'category' in labels:
            cat_preds = torch.argmax(predictions['category'], dim=1).cpu()
            cat_true = labels['category'].cpu()
            
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(cat_true, cat_preds)
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.category_labels, yticklabels=self.category_labels)
            plt.title('Category Confusion Matrix')
            plt.savefig(f'{save_path}/category_cm.png')
            plt.close()
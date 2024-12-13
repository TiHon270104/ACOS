import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from utils.data import ABSADataset
from models.base_model import ABSAModel
from utils.loss import ABSALoss
from utils.metrics import Metrics
from config import config
import os
from tqdm import tqdm

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def move_batch_to_device(batch, device):
    """Move batch data to device (CPU/GPU)"""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

def train():
    # Set seed
    set_seed(config['seed'])
    
    # Create save directory
    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = ABSADataset(config['train_path'])
    dev_dataset = ABSADataset(config['dev_path'])
    test_dataset = ABSADataset(config['test_path'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=config['batch_size']
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size']
    )
    
    # Initialize model, criterion, optimizer
    print(f"Initializing model... Using device: {config['device']}")
    model = ABSAModel(config).to(config['device'])
    criterion = ABSALoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate']
    )
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    # Initialize metrics
    metrics = Metrics()
    best_f1 = 0
    
    # Training loop
    print("Starting training...")
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        
        # Progress bar for training
        train_iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]",
            total=len(train_loader)
        )
        
        for batch in train_iterator:
            # Move batch to device
            batch = move_batch_to_device(batch, config['device'])
            
            # Forward pass
            outputs = model(
                batch['input_ids'],
                batch['attention_mask']
            )
            
            # Calculate loss
            loss, loss_items = criterion(outputs, batch)
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['max_grad_norm']
            )
            
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            train_iterator.set_postfix({
                'loss': f"{loss.item():.4f}"
            })
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluate on dev set
        print("\nEvaluating on dev set...")
        model.eval()
        dev_metrics = evaluate(model, dev_loader, metrics, config['device'])
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print("Dev Metrics:")
        for metric_name, value in dev_metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        # Save best model
        if dev_metrics['aspect_f1'] > best_f1:
            best_f1 = dev_metrics['aspect_f1']
            print(f"\nNew best F1: {best_f1:.4f}")
            print("Saving best model...")
            torch.save(
                model.state_dict(),
                f"{config['save_path']}/best_model.pt"
            )
    
    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    model.load_state_dict(
        torch.load(f"{config['save_path']}/best_model.pt")
    )
    test_metrics = evaluate(model, test_loader, metrics, config['device'])
    
    # Print final test metrics
    print("\nTest Metrics:")
    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Plot confusion matrices
    print("\nGenerating confusion matrices...")
    model.eval()
    with torch.no_grad():
        all_predictions = {}
        all_labels = {}
        
        for batch in tqdm(dev_loader, desc="Processing dev set"):
            batch = move_batch_to_device(batch, config['device'])
            outputs = model(batch['input_ids'], batch['attention_mask'])
            
            for key in outputs:
                if key not in all_predictions:
                    all_predictions[key] = []
                all_predictions[key].append(outputs[key].cpu())
                
            for key in batch:
                if key not in all_labels:
                    all_labels[key] = []
                if isinstance(batch[key], torch.Tensor):
                    all_labels[key].append(batch[key].cpu())
        
        # Concatenate all batches
        for key in all_predictions:
            all_predictions[key] = torch.cat(all_predictions[key], dim=0)
        for key in all_labels:
            if all_labels[key]:
                all_labels[key] = torch.cat(all_labels[key], dim=0)
                
        metrics.plot_confusion_matrices(
            all_predictions, 
            all_labels, 
            config['save_path']
        )
    
    print(f"\nTraining completed. Results saved in {config['save_path']}")

def evaluate(model, dataloader, metrics, device):
    """Evaluate model on given dataloader"""
    model.eval()
    all_predictions = {}
    all_labels = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = move_batch_to_device(batch, device)
            outputs = model(batch['input_ids'], batch['attention_mask'])
            
            for key in outputs:
                if key not in all_predictions:
                    all_predictions[key] = []
                all_predictions[key].append(outputs[key].cpu())
                
            for key in batch:
                if key not in all_labels:
                    all_labels[key] = []
                if isinstance(batch[key], torch.Tensor):
                    all_labels[key].append(batch[key].cpu())
    
    # Concatenate all batches
    for key in all_predictions:
        all_predictions[key] = torch.cat(all_predictions[key], dim=0)
    for key in all_labels:
        if all_labels[key]:
            all_labels[key] = torch.cat(all_labels[key], dim=0)
            
    return metrics.compute_metrics(all_predictions, all_labels)

if __name__ == '__main__':
    train()
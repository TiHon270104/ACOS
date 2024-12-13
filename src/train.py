import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from utils.data import ABSADataset
from models.base_model import ABSAModel
from utils.loss import ABSALoss
from utils.metrics import Metrics
from config import config
import os
from tqdm import tqdm

# Enable cuDNN benchmarking
cudnn.benchmark = True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train():
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Set seed
    set_seed(config['seed'])
    
    # Create save directory
    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])
    
    # Load datasets with optimized DataLoader
    print("Loading datasets...")
    train_dataset = ABSADataset(config['train_path'])
    dev_dataset = ABSADataset(config['dev_path'])
    test_dataset = ABSADataset(config['test_path'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Initialize model, criterion, optimizer
    print(f"Initializing model... Using device: {config['device']}")
    model = ABSAModel(config).to(config['device'])
    criterion = ABSALoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate']
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if config['fp16'] else None
    
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
        optimizer.zero_grad()
        
        train_iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]"
        )
        
        for i, batch in enumerate(train_iterator):
            # Move batch to device
            batch = {
                k: v.to(config['device']) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Mixed precision training
            with autocast(enabled=config['fp16']):
                outputs = model(batch['input_ids'], batch['attention_mask'])
                loss, loss_items = criterion(outputs, batch)
                loss = loss / config['gradient_accumulation_steps']
            
            if config['fp16']:
                scaler.scale(loss).backward()
                if (i + 1) % config['gradient_accumulation_steps'] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config['max_grad_norm']
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (i + 1) % config['gradient_accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config['max_grad_norm']
                    )
                    optimizer.step()
                    optimizer.zero_grad()
            
            scheduler.step()
            total_loss += loss.item() * config['gradient_accumulation_steps']
            
            # Update progress bar
            train_iterator.set_postfix({
                'loss': f"{loss.item():.4f}"
            })
        
        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate on dev set
        print("\nEvaluating on dev set...")
        dev_metrics = evaluate(model, dev_loader, metrics, config['device'])
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"Average Training Loss: {avg_loss:.4f}")
        print("Dev Metrics:", dev_metrics)
        
        # Save best model
        if dev_metrics['aspect_f1'] > best_f1:
            best_f1 = dev_metrics['aspect_f1']
            print(f"New best F1: {best_f1:.4f}")
            torch.save(
                model.state_dict(),
                f"{config['save_path']}/best_model.pt"
            )
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(f"{config['save_path']}/best_model.pt"))
    test_metrics = evaluate(model, test_loader, metrics, config['device'])
    print("Test Metrics:", test_metrics)

if __name__ == '__main__':
    train()
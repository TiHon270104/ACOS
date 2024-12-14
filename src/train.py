import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from models import ACOSModel
from utils import ASODataset, ACOSLoss, ACOSMetrics
from config import config

def train():
    # Load data
    train_dataset = ASODataset(config['train_path'])
    dev_dataset = ASODataset(config['dev_path'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config['batch_size']
    )
    
    # Initialize model
    model = ACOSModel(config)
    model.cuda()
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate']
    )
    
    # Initialize scheduler
    num_training_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    # Initialize loss & metrics
    criterion = ACOSLoss(config)
    metrics = ACOSMetrics()
    
    # Training loop
    best_f1 = 0
    for epoch in range(config['num_epochs']):
        model.train()
        for batch in tqdm(train_loader):
            # Move batch to GPU
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                batch['input_ids'],
                batch['attention_mask']
            )
            
            # Calculate loss
            loss_dict = criterion.compute_loss(outputs, batch)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Update weights
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['max_grad_norm']
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        # Evaluation
        model.eval()
        metrics.reset()
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(
                    batch['input_ids'],
                    batch['attention_mask']
                )
                metrics.update(outputs, batch)
                
        # Calculate metrics
        results = metrics.compute()
        
        # Save best model
        if results['aspect_f1'] > best_f1:
            best_f1 = results['aspect_f1']
            torch.save(
                model.state_dict(),
                f"{config['save_dir']}/best_model.pt"
            )
            
        print(f"Epoch {epoch+1}:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    train()
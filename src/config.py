import torch

config = {
    # Model configs
    'model_name': 'vinai/phobert-base',
    'hidden_size': 768,
    'num_attention_heads': 8,
    'max_length': 256,
    'dropout': 0.1,
    
    # Training configs
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 10,
    'warmup_steps': 1000,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    
    # Loss weights
    'aspect_loss_weight': 1.0,
    'opinion_loss_weight': 1.0,
    'pairing_loss_weight': 1.0,
    'sentiment_loss_weight': 1.0,
    'category_loss_weight': 1.0,
    
    # Labels
    'num_sentiments': 3,  # positive, negative, neutral
    'num_categories': 5,  # APPEARANCE, TECHNICAL, SPECIALIZE, OTHER, CHARACTERISTIC
    
    # Paths
    'train_path': 'data/train.csv',
    'dev_path': 'data/dev.csv',
    'test_path': 'data/test.csv',
    'save_dir': 'checkpoints'
}
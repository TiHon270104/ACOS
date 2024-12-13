import torch

config = {
    # Model configs
    'model_name': 'vinai/phobert-base',
    'hidden_size': 768,
    'max_length': 96,  # Giảm max length
    
    # Training configs
    'batch_size': 64,  # Tăng batch size
    'learning_rate': 2e-5,
    'num_epochs': 10,
    'warmup_steps': 1000,
    'max_grad_norm': 1.0,
    'gradient_accumulation_steps': 2,
    
    # Optimization
    'fp16': True,  # Mixed precision training
    'num_workers': 4,  # DataLoader workers
    
    # Model architecture
    'freeze_embeddings': True,  # Freeze embedding layer
    'lstm_layers': 1,  # Giảm số LSTM layers
    'dropout': 0.1,
    
    # Labels
    'sentiment_labels': ['positive', 'negative', 'neutral'],
    'category_labels': ['APPEARANCE', 'TECHNICAL', 'SPECIALIZE', 'OTHER', 'CHARACTERISTIC'],
    
    # Paths
    'train_path': 'ACOS/src/data/train.csv',
    'dev_path': 'ACOS/src/data/dev.csv',
    'test_path': 'ACOS/src/data/test.csv',
    'save_path': 'results',
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Random seed
    'seed': 42
}
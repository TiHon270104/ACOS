import torch

config = {
    # Model configs
    'model_name': 'vinai/phobert-base',
    'hidden_size': 768,
    'max_length': 128,
    
    # Training configs
    'batch_size': 32,
    'learning_rate': 2e-5,
    'num_epochs': 10,
    'warmup_steps': 1000,
    'max_grad_norm': 1.0,
    
    # Labels
    'sentiment_labels': ['positive', 'negative', 'neutral'],
    'category_labels': ['APPEARANCE', 'TECHNICAL', 'SPECIALIZE', 'OTHER', 'CHARACTERISTIC'],
    
    # Paths
    'train_path': 'ACOS/src/data/train.csv',  # Sửa đường dẫn
    'dev_path': 'ACOS/src/data/dev.csv',      # Sửa đường dẫn  
    'test_path': 'ACOS/src/data/test.csv',    # Sửa đường dẫn
    'save_path': 'results',
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Random seed
    'seed': 42
}
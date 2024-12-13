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
    'train_path': 'data/train.csv',
    'dev_path': 'data/dev.csv',
    'test_path': 'data/test.csv',
    'save_path': 'results',
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Random seed
    'seed': 42
}
def get_baseline_config():
    """Define the baseline configuration for experiments"""
    return {
        'name': 'baseline',
        'image_type': 'W',                # 'W' for broadband, 'B' for fluorescent
        'backbone': 'resnet34',           # Base backbone
        'use_attention': False,           # No attention mechanisms
        'batch_size': 4,                  # Standard batch size
        'img_size': (256, 256),
        'num_epochs': 2,                 # Sufficient training time
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'pretrained': True,               # Use pretrained weights
        'save_dir': 'experiments',
        'seed': 42,                       # Fixed seed for reproducibility
        'optimizer': 'adam',
        'visualize_every': 5,
        'save_visualizations': True,
        'save_model': True,
        'loss_fn': 'combo',               # Default combo loss
        'loss_alpha': 0.5,                # Default loss parameter
    }

import os
import torch
import json
import copy
from datetime import datetime
import pandas as pd
from baseline_config import get_baseline_config

# Import your modules
from dataset import prepare_augmented_data
from advanced_models import UNetWithBackbone
from losses import get_loss_function
from train import train_model
from cross_validation import cross_validate
from utils import ensure_dir, get_device
from visualize import visualize_predictions

def run_single_experiment(experiment_name, modified_params, cv=False):
    """
    Run a single experiment with modified parameters from baseline
    
    Args:
        experiment_name: Name for this experiment
        modified_params: Dictionary of parameters to change from baseline
        cv: Whether to use cross-validation (default: False)
    """
    # Get baseline config and update with modified parameters
    config = get_baseline_config()
    
    # Update with modified parameters
    for key, value in modified_params.items():
        config[key] = value
    
    # Add experiment name
    config['name'] = experiment_name
    
    # Create timestamp for the experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Update save directory with timestamp
    save_dir = f"experiments/{experiment_name}_{timestamp}"
    config['save_dir'] = save_dir
    ensure_dir(save_dir)
    
    # Save the configuration
    with open(f"{save_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Starting experiment: {experiment_name}")
    print(f"Modified parameters: {modified_params}")
    print(f"Results will be saved to: {save_dir}")
    
    # Prepare data
    data_dir = 'manual_labels'  # Update this to your data path
    train_loader, test_loader = prepare_augmented_data(
        data_dir=data_dir,
        image_type=config['image_type'],
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        seed=config['seed']
    )
    
    # Create model
    model = UNetWithBackbone(
        n_classes=1,
        backbone=config['backbone'],
        pretrained=config['pretrained'],
        use_attention=config['use_attention']
    )
    
    # Get device
    device = get_device()
    model = model.to(device)
    
    # Create loss function
    criterion = get_loss_function(config)
    
    # Create optimizer
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config['learning_rate'],
            momentum=0.9,
            weight_decay=config['weight_decay']
        )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, threshold=0.01, min_lr=1e-6
    )
    
    # Train the model or perform cross-validation
    if cv:
        # Number of folds
        n_splits = 5
        
        # Run cross-validation
        results = cross_validate(
            model_class=UNetWithBackbone,
            config=config,
            data_dir=data_dir,
            n_splits=n_splits
        )
        
        # Save cross-validation results
        torch.save(results, f"{save_dir}/cv_results.pth")
        
        # Return mean IoU and standard deviation
        return {
            'experiment': experiment_name,
            'iou': results['cv_summary']['iou_mean'],
            'iou_std': results['cv_summary']['iou_std'],
            'config': config,
            'results': results
        }
    else:
        # Train the model
        results = train_model(
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config['num_epochs'],
            device=device,
            config=config
        )
        
        # Save full training results
        torch.save({
            'config': config,
            'model_state': model.state_dict(),
            'metrics': results
        }, f"{save_dir}/training_results.pth")
        
        # Visualize predictions
        visualize_predictions(
            model, test_loader, device, num_samples=5,
            save_path=f"{save_dir}/predictions.png"
        )
        
        # Return best IoU
        return {
            'experiment': experiment_name,
            'iou': results['best_iou'],
            'config': config,
            'results': results
        }

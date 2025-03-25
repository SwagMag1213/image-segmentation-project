import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Import from our modular framework
from dataset import prepare_augmented_data
from advanced_models import UNetWithBackbone
from losses import get_loss_function
from train import train_model
from cross_validation import cross_validate
from utils import ensure_dir, get_device
from visualize import visualize_predictions
from wnet import WNet  # <- NEW: W-Net

def get_config_unet():
    """Configuration for U-Net (supervised)"""
    return {
        'name': 'unet_resnet34',
        'model_type': 'unet',
        'image_type': 'W',
        'backbone': 'resnet34',
        'use_attention': False,
        'batch_size': 4,
        'img_size': (256, 256),
        'num_epochs': 5,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'pretrained': True,
        'save_dir': 'experiments/unet',
        'seed': 42,
        'optimizer': 'adam',
        'visualize_every': 5,
        'save_visualizations': True,
        'save_model': True,
        'loss_fn': 'focal',
        'loss_alpha': 0.25,
        'focal_gamma': 2.0,
    }

def get_config_wnet_semi():
    """Configuration for W-Net (semi-supervised: seg + recon)"""
    config = get_config_unet()
    config.update({
        'name': 'wnet_semi',
        'model_type': 'wnet',
        'save_dir': 'experiments/wnet_semi',
    })
    return config

def get_config_wnet_unsupervised():
    """Configuration for W-Net (unsupervised: NCut + recon)"""
    config = get_config_unet()
    config.update({
        'name': 'wnet_unsupervised',
        'model_type': 'wnet',
        'loss_fn': 'softncut',
        'save_dir': 'experiments/wnet_unsupervised',
        'use_ncut': True,
        'ncut_weight': 1.0,
        'recon_weight': 1.0,
    })
    return config


def main():
    # Create timestamp for the experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get configuration

    #config = get_config_unet()
    #config = get_config_wnet_semi()
    config = get_config_wnet_unsupervised()
    
    data_dir = 'data/manual_labels'  # Update this path to your data directory
    
    # Update save directories with timestamp
    save_dir = f"{config['save_dir']}_{timestamp}"
    config['save_dir'] = save_dir
    ensure_dir(save_dir)
    
    print(f"Starting experiment: {config['name']}")
    print(f"Results will be saved to: {save_dir}")
    
    # PART 1: Train on full dataset
    print("\n=== Training on Full Dataset ===\n")
    
    # Prepare data
    train_loader, test_loader = prepare_augmented_data(
        data_dir=data_dir,
        image_type=config['image_type'],
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        seed=config['seed']
    )
    
    # Create model
    if config['model_type'] == 'wnet':
        model = WNet(
            n_channels=1,
            n_classes=1,
            backbone=config['backbone'],
            pretrained=config['pretrained'],
            use_attention=config['use_attention']
        )
    else:
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
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, threshold=0.01, min_lr=1e-6
    )
    
    # Optional second loss for W-Net
    criterion_recon = None
    if config['model_type'] == 'wnet':
        import torch.nn as nn
        criterion_recon = nn.MSELoss()          

    # Train the model
    full_train_result = train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['num_epochs'],
        device=device,
        config=config,
        criterion_recon=criterion_recon,  # Pass reconstruction loss for W-Net
    )
    
    # Save full training results
    torch.save({
        'config': config,
        'model_state': model.state_dict(),
        'metrics': full_train_result
    }, f"{save_dir}/full_training_results.pth")
    
    # Visualize predictions
    visualize_predictions(
        model, test_loader, device, num_samples=5
    )
    
    # PART 2: Perform cross-validation with the same configuration
    print("\n=== Performing Cross-Validation ===\n")
    
    # Number of folds
    n_splits = 5
    
    # Run cross-validation
    cv_results = cross_validate(
        model_class=UNetWithBackbone,
        config=config,
        data_dir=data_dir,
        n_splits=n_splits
    )
    
    # Save cross-validation results
    torch.save(cv_results, f"{save_dir}/cv_results.pth")
    
    # Compare full training vs cross-validation
    full_train_iou = full_train_result['test_metrics'][-1]['iou']
    cv_mean_iou = cv_results['cv_summary']['iou_mean']
    cv_std_iou = cv_results['cv_summary']['iou_std']
    
    print("\n=== Results Comparison ===")
    print(f"Full Training IoU: {full_train_iou:.4f}")
    print(f"Cross-Validation IoU: {cv_mean_iou:.4f} ± {cv_std_iou:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Plot IoU comparison
    plt.bar([0, 1], [full_train_iou, cv_mean_iou], 
            yerr=[0, cv_std_iou], capsize=10, color=['blue', 'green'])
    
    plt.xticks([0, 1], ['Full Training', 'Cross-Validation'])
    plt.ylabel('IoU Score')
    plt.title(f'{config["name"]} - Performance Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels
    plt.text(0, full_train_iou/2, f"{full_train_iou:.4f}", 
             ha='center', va='center', color='white', fontweight='bold')
    plt.text(1, cv_mean_iou/2, f"{cv_mean_iou:.4f} ± {cv_std_iou:.4f}", 
             ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison.png", dpi=200)
    plt.show()
    
    print(f"\nExperiment completed. All results saved to {save_dir}")

if __name__ == "__main__":
    main()
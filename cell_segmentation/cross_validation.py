import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GroupKFold
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from collections import defaultdict
import copy
import time

from dataset import CellSegmentationDataset
from losses import get_loss_function
from utils import get_device, calculate_metrics
from visualize import visualize_predictions

def prepare_cross_validation_data(data_dir, image_type, n_splits=5, img_size=(256, 256), seed=42):
    """
    Prepare data for cross-validation while preventing data leakage
    
    Args:
        data_dir: Directory containing the data
        image_type: 'B' for fluorescent, 'W' for broadband
        n_splits: Number of folds for cross-validation
        img_size: Image size for resizing
        seed: Random seed for reproducibility
    
    Returns:
        dataset: The complete dataset
        fold_indices: List of (train_indices, val_indices) tuples for each fold
        image_groups: Dictionary mapping base image names to indices
    """
    # Paths to augmented directories
    aug_dir = os.path.join(data_dir, f"augmented_{image_type}")
    aug_images_dir = os.path.join(aug_dir, "images")
    aug_masks_dir = os.path.join(aug_dir, "masks")
    
    # Verify that the augmented directories exist
    if not os.path.exists(aug_dir):
        raise FileNotFoundError(f"Augmented directory {aug_dir} not found.")
    
    # Get all files from augmented dataset
    all_images = sorted(os.listdir(aug_images_dir))
    
    # Store all image and mask paths
    image_paths = []
    mask_paths = []
    groups = []  # Original image group for each augmented image
    
    # Dictionary to store indices by base name
    image_groups = defaultdict(list)
    
    # Process all images
    for i, img in enumerate(all_images):
        # Extract base name (part before _orig or _aug)
        if "_orig.tif" in img:
            base_name = img.split('_orig.tif')[0]
        elif "_aug" in img:
            # Extract the base name from augmented images (everything before _aug)
            base_name = img.split('_aug')[0]
        else:
            # Skip if not following expected naming convention
            continue
        
        # Check if corresponding mask exists
        mask_path = os.path.join(aug_masks_dir, img)
        if not os.path.exists(mask_path):
            continue
        
        # Add to dataset
        image_paths.append(os.path.join(aug_images_dir, img))
        mask_paths.append(mask_path)
        groups.append(base_name)
        
        # Store index by base name
        image_groups[base_name].append(len(image_paths) - 1)
    
    # Create the dataset
    dataset = CellSegmentationDataset(image_paths, mask_paths, img_size=img_size)
    
    # Set up GroupKFold to ensure augmentations from the same original image stay together
    group_kfold = GroupKFold(n_splits=n_splits)
    fold_indices = list(group_kfold.split(np.arange(len(image_paths)), groups=groups))
    
    print(f"Prepared dataset with {len(image_paths)} images from {len(image_groups)} base images")
    print(f"Split into {n_splits} folds for cross-validation")
    
    return dataset, fold_indices, image_groups

def cross_validate(model_class, config, data_dir, n_splits=5):
    """
    Perform cross-validation on a model with the given configuration
    
    Args:
        model_class: Class for model initialization (e.g., UNetWithBackbone)
        config: Configuration dictionary
        data_dir: Directory containing the data
        n_splits: Number of folds for cross-validation
    
    Returns:
        Dictionary with cross-validation results
    """
    print(f"Starting {n_splits}-fold cross-validation for {config['name']}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device setup
    device = get_device()
    
    # Prepare dataset for cross-validation
    dataset, fold_indices, image_groups = prepare_cross_validation_data(
        data_dir=data_dir,
        image_type=config['image_type'],
        n_splits=n_splits,
        img_size=config['img_size'],
        seed=config['seed']
    )
    
    # Storage for fold results
    fold_results = []
    
    # Metrics across all folds
    all_val_metrics = defaultdict(list)
    best_models = []
    
    # For each fold
    for fold_idx, (train_indices, val_indices) in enumerate(fold_indices):
        print(f"\n----- Fold {fold_idx+1}/{n_splits} -----")
        
        # Create train and validation subsets
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset, 
            batch_size=config['batch_size'], 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_subset, 
            batch_size=config['batch_size'], 
            shuffle=False
        )
        
        # Initialize model
        model = model_class(
            n_classes=1, 
            backbone=config['backbone'],
            pretrained=config['pretrained'],
            use_attention=config['use_attention']
        )
        model = model.to(device)
        
        # Initialize loss function
        criterion = get_loss_function(config)
        
        # Initialize optimizer
        if config.get('optimizer', 'adam') == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=config['learning_rate'], 
                weight_decay=config.get('weight_decay', 1e-5)
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=config['learning_rate'], 
                momentum=0.9,
                weight_decay=config.get('weight_decay', 1e-5)
            )
        
        # Initialize learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, threshold=0.01, min_lr=1e-6
        )
        
        # Setup training tracking
        train_metrics_history = []
        val_metrics_history = []
        best_iou = 0.0
        best_model_state = None
        best_epoch = 0
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(config['num_epochs']):
            # Train one epoch
            model.train()
            epoch_loss = 0
            train_metrics = defaultdict(float)
            num_samples = 0
            
            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    batch_metrics = calculate_metrics(torch.sigmoid(outputs), masks)
                    batch_size = images.size(0)
                    
                    for k, v in batch_metrics.items():
                        train_metrics[k] += v * batch_size
                    
                    num_samples += batch_size
                    epoch_loss += loss.item() * batch_size
            
            # Normalize train metrics
            epoch_loss /= num_samples
            for k in train_metrics:
                train_metrics[k] /= num_samples
            
            train_metrics['loss'] = epoch_loss
            train_metrics_history.append(train_metrics)
            
            # Evaluate on validation set
            model.eval()
            val_metrics = defaultdict(float)
            val_samples = 0
            
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    outputs = model(images)
                    
                    # Calculate loss
                    loss = criterion(outputs, masks)
                    batch_size = images.size(0)
                    val_metrics['loss'] += loss.item() * batch_size
                    
                    # Calculate other metrics
                    batch_metrics = calculate_metrics(torch.sigmoid(outputs), masks)
                    for k, v in batch_metrics.items():
                        val_metrics[k] += v * batch_size
                    
                    val_samples += batch_size
            
            # Normalize validation metrics
            for k in val_metrics:
                val_metrics[k] /= val_samples
            
            val_metrics_history.append(val_metrics)
            
            # Update learning rate
            scheduler.step(val_metrics['iou'])
            
            # Print progress
            print(f"Epoch {epoch+1}/{config['num_epochs']} - "
                  f"Train loss: {train_metrics['loss']:.4f}, "
                  f"Val IoU: {val_metrics['iou']:.4f}")
            
            # Save best model
            if val_metrics['iou'] > best_iou:
                best_iou = val_metrics['iou']
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                print(f"Saved new best model with IoU: {best_iou:.4f}")
        
        # Training complete for this fold
        time_elapsed = time.time() - start_time
        print(f"Fold {fold_idx+1} training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best IoU: {best_iou:.4f} at epoch {best_epoch+1}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        final_metrics = defaultdict(float)
        val_samples = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                
                # Calculate metrics
                batch_metrics = calculate_metrics(torch.sigmoid(outputs), masks)
                batch_size = images.size(0)
                
                for k, v in batch_metrics.items():
                    final_metrics[k] += v * batch_size
                
                val_samples += batch_size
        
        # Normalize final metrics
        for k in final_metrics:
            final_metrics[k] /= val_samples
        
        print(f"Final metrics for fold {fold_idx+1}: {final_metrics}")
        
        # Save fold results
        fold_result = {
            'fold': fold_idx + 1,
            'train_metrics': train_metrics_history,
            'val_metrics': val_metrics_history,
            'best_iou': best_iou,
            'best_epoch': best_epoch,
            'final_metrics': final_metrics,
            'training_time': time_elapsed
        }
        
        # Append to results list
        fold_results.append(fold_result)
        best_models.append(best_model_state)
        
        # Collect metrics for summary
        for metric, value in final_metrics.items():
            all_val_metrics[metric].append(value)
        
        # Create save directory if requested
        if config.get('save_model', False):
            save_dir = config.get('save_dir', 'experiments')
            os.makedirs(save_dir, exist_ok=True)
            torch.save(best_model_state, f"{save_dir}/{config['name']}_fold{fold_idx+1}_best.pth")
    
    # Calculate mean and std of metrics across folds
    cv_summary = {}
    for metric, values in all_val_metrics.items():
        cv_summary[f'{metric}_mean'] = np.mean(values)
        cv_summary[f'{metric}_std'] = np.std(values)
    
    print("\n----- Cross-Validation Summary -----")
    for metric, mean_value in cv_summary.items():
        if metric.endswith('_mean'):
            base_metric = metric[:-5]  # Remove '_mean'
            std_value = cv_summary[f'{base_metric}_std']
            print(f"{base_metric}: {mean_value:.4f} ± {std_value:.4f}")
    
    # Create plots
    plot_cross_validation_results(fold_results, config)
    
    # Return results
    cv_results = {
        'config': config,
        'fold_results': fold_results,
        'cv_summary': cv_summary,
        'best_models': best_models
    }
    
    return cv_results

def plot_cross_validation_results(fold_results, config):
    """Plot metrics across folds"""
    n_folds = len(fold_results)
    
    plt.figure(figsize=(15, 10))
    
    # Plot training and validation IoU for each fold
    plt.subplot(2, 2, 1)
    for fold_idx, result in enumerate(fold_results):
        plt.plot([m['iou'] for m in result['train_metrics']], 
                '--', label=f'Fold {fold_idx+1} Train')
        plt.plot([m['iou'] for m in result['val_metrics']], 
                '-', label=f'Fold {fold_idx+1} Val')
    
    plt.title('IoU Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.grid(True)
    plt.legend()
    
    # Plot training and validation loss for each fold
    plt.subplot(2, 2, 2)
    for fold_idx, result in enumerate(fold_results):
        plt.plot([m['loss'] for m in result['train_metrics']], 
                '--', label=f'Fold {fold_idx+1} Train')
        plt.plot([m['loss'] for m in result['val_metrics']], 
                '-', label=f'Fold {fold_idx+1} Val')
    
    plt.title('Loss Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Bar chart of best IoU for each fold
    plt.subplot(2, 2, 3)
    best_ious = [result['best_iou'] for result in fold_results]
    fold_numbers = [f'Fold {i+1}' for i in range(n_folds)]
    
    plt.bar(fold_numbers, best_ious)
    plt.title('Best IoU by Fold')
    plt.ylabel('IoU')
    plt.axhline(y=np.mean(best_ious), color='r', linestyle='-', label=f'Mean: {np.mean(best_ious):.4f}')
    plt.legend()
    
    # Bar chart of final metrics across folds
    plt.subplot(2, 2, 4)
    metrics = ['iou', 'f1', 'precision', 'recall']
    means = []
    stds = []
    
    for metric in metrics:
        values = [result['final_metrics'][metric] for result in fold_results]
        means.append(np.mean(values))
        stds.append(np.std(values))
    
    x = np.arange(len(metrics))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, metrics)
    plt.title('Final Metrics (Mean ± Std)')
    plt.ylabel('Value')
    
    plt.tight_layout()
    
    # Save figure if requested
    if config.get('save_visualizations', False):
        save_dir = config.get('save_dir', 'experiments')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{config['name']}_cv_results.png", dpi=200, bbox_inches='tight')
    
    plt.show()
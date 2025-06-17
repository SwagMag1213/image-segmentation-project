"""
Simplified Training Module for Supervised Cell Segmentation
Clean, focused training functions without unsupervised methods
"""

import time
import torch
import copy
from collections import defaultdict
from typing import Dict, Optional
import matplotlib.pyplot as plt

from dataset import prepare_data, CellAugmenter
from utils import calculate_metrics, EarlyStopping


def train_epoch(model: torch.nn.Module, loader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, 
                device: torch.device) -> Dict:
    """
    Train one epoch.
    
    Args:
        model: Neural network model
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Computation device
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    epoch_loss = 0
    metrics = defaultdict(float)
    num_samples = 0
    
    for images, masks in loader:
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
                metrics[k] += v * batch_size
            
            num_samples += batch_size
            epoch_loss += loss.item() * batch_size
    
    # Normalize metrics
    epoch_loss /= num_samples
    for k in metrics:
        metrics[k] /= num_samples
    
    metrics['loss'] = epoch_loss
    return metrics


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, 
             device: torch.device, criterion: torch.nn.Module) -> Dict:
    """
    Evaluate the model on the given data loader.
    
    Args:
        model: Neural network model
        loader: Evaluation data loader
        device: Computation device
        criterion: Loss function
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    metrics = defaultdict(float)
    num_samples = 0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            batch_metrics = calculate_metrics(torch.sigmoid(outputs), masks)
            batch_size = images.size(0)
            
            for k, v in batch_metrics.items():
                metrics[k] += v * batch_size
            
            metrics['loss'] += loss.item() * batch_size
            num_samples += batch_size
    
    # Normalize metrics
    for k in metrics:
        metrics[k] /= num_samples
    
    return metrics


def train_model(model: torch.nn.Module, train_images: list, train_masks: list,
                val_images: list, val_masks: list, criterion: torch.nn.Module,
                optimizer: torch.optim.Optimizer, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                num_epochs: int, device: torch.device, config: Dict,
                augmentations_per_image: int = 0, save_plots: bool = True) -> Dict:
    """
    Complete training loop with validation, augmentation, and tracking.
    
    Args:
        model: Neural network model
        train_images: List of training image paths
        train_masks: List of training mask paths
        val_images: List of validation image paths
        val_masks: List of validation mask paths
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of training epochs
        device: Computation device
        config: Configuration dictionary
        augmentations_per_image: Number of augmentations per image (0 = no augmentation)
        save_plots: Whether to save training plots
        
    Returns:
        Dictionary with training history and results
    """
    augmenter = CellAugmenter(augmentations_per_image)
    # Apply augmentation to training data if requested
    if augmentations_per_image > 0:
        aug_train_images, aug_train_masks = augmenter.augment_training_data(
            train_images, train_masks
        )
    else:
        aug_train_images, aug_train_masks = train_images, train_masks
    
    train_loader = prepare_data(
        aug_train_images, aug_train_masks,
        config['batch_size'], config['img_size'], shuffle=True
    )
    val_loader = prepare_data(
        val_images, val_masks,  # No augmentation for validation
        config['batch_size'], config['img_size'], shuffle=False
    )
    
    # Setup tracking
    train_metrics_history = []
    val_metrics_history = []
    lr_history = []
    best_iou = 0.0
    best_model_state = None
    best_epoch = 0
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 7),
        min_delta=config.get('early_stopping_min_delta', 0.001)
    )
    
    if config.get('verbose', True):
        print("Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Train one epoch
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        train_metrics_history.append(train_metrics)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, criterion)
        val_metrics_history.append(val_metrics)
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['iou'])
            else:
                scheduler.step()
        
        # Record learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # Print progress (only if verbose)
        if config.get('verbose', True):
            print(f"Epoch {epoch+1:3d}/{num_epochs} - "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train IoU: {train_metrics['iou']:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val IoU: {val_metrics['iou']:.4f}, "
                  f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        
        # Early stopping
        if early_stopping.step(val_metrics['iou']):
            if config.get('verbose', True):
                print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Training complete
    time_elapsed = time.time() - start_time
    if config.get('verbose', True):
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best validation IoU: {best_iou:.4f} at epoch {best_epoch+1}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Create training plots
    if save_plots and config.get('save_plots', False):
        plot_training_history(train_metrics_history, val_metrics_history, lr_history, config)
    
    augmenter.cleanup()  # Clean up temporary files if any
    
    return {
        'train_metrics': train_metrics_history,
        'val_metrics': val_metrics_history,
        'lr_history': lr_history,
        'best_iou': best_iou,
        'best_epoch': best_epoch,
        'best_model_state': best_model_state,
        'training_time': time_elapsed,
        'final_train_metrics': train_metrics,
        'final_val_metrics': val_metrics
    }


def plot_training_history(train_metrics: list, val_metrics: list, lr_history: list, 
                         config: Dict) -> None:
    """
    Plot training history curves.
    
    Args:
        train_metrics: List of training metrics per epoch
        val_metrics: List of validation metrics per epoch
        lr_history: List of learning rates per epoch
        config: Configuration dictionary
    """
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot([m['loss'] for m in train_metrics], label='Train', linewidth=2)
    plt.plot([m['loss'] for m in val_metrics], label='Validation', linewidth=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # IoU plot
    plt.subplot(1, 3, 2)
    plt.plot([m['iou'] for m in train_metrics], label='Train', linewidth=2)
    plt.plot([m['iou'] for m in val_metrics], label='Validation', linewidth=2)
    plt.title('Training and Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate plot
    plt.subplot(1, 3, 3)
    plt.plot(lr_history, linewidth=2, color='orange')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if save directory is specified
    if 'save_dir' in config and config['save_dir']:
        import os
        os.makedirs(config['save_dir'], exist_ok=True)
        plt.savefig(f"{config['save_dir']}/training_curves.png", dpi=200, bbox_inches='tight')
        print(f"Training plots saved to {config['save_dir']}/training_curves.png")
    
    plt.show()


def quick_train(model: torch.nn.Module, train_images: list, train_masks: list,
                val_images: list, val_masks: list, config: Dict,
                device: torch.device = None, augmentations_per_image: int = 0) -> Dict:
    """
    Quick training function with standard setup.
    
    Args:
        model: Neural network model
        train_images: List of training image paths
        train_masks: List of training mask paths
        val_images: List of validation image paths
        val_masks: List of validation mask paths
        config: Configuration dictionary
        device: Computation device (auto-detected if None)
        augmentations_per_image: Number of augmentations per image
        
    Returns:
        Dictionary with training results
    """
    if device is None:
        from utils import get_device
        device = get_device()
    
    model = model.to(device)
    
    # Setup loss function
    from losses import get_loss_function
    criterion = get_loss_function(config)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=config.get('scheduler_patience', 3),
        threshold=0.01,
        min_lr=1e-6
    )
    
    # Train model
    results = train_model(
        model=model,
        train_images=train_images,
        train_masks=train_masks,
        val_images=val_images,
        val_masks=val_masks,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.get('num_epochs', 50),
        device=device,
        config=config,
        augmentations_per_image=augmentations_per_image,
        save_plots=config.get('save_plots', True)
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Training module loaded successfully!")
    print("Use train_model() for full control or quick_train() for standard setup")
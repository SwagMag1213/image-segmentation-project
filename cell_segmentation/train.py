import time
import torch
from tqdm.auto import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import copy
from utils import calculate_metrics

def train_epoch(model, loader, optimizer, criterion, device):
    """Train the model for one epoch"""
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

def evaluate(model, loader, device, criterion=None):
    """Evaluate the model on the given data loader"""
    model.eval()
    metrics = defaultdict(float)
    num_samples = 0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # Calculate loss if criterion is provided
            if criterion is not None:
                loss = criterion(outputs, masks)
                batch_size = images.size(0)
                metrics['loss'] += loss.item() * batch_size
            
            # Calculate metrics
            batch_metrics = calculate_metrics(torch.sigmoid(outputs), masks)
            batch_size = images.size(0)
            
            for k, v in batch_metrics.items():
                metrics[k] += v * batch_size
            
            num_samples += batch_size
    
    # Normalize metrics
    for k in metrics:
        metrics[k] /= num_samples
    
    return metrics

def train_model(train_loader, test_loader, model, criterion, optimizer, scheduler, num_epochs, device, config):
    """
    Train and evaluate the model
    
    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        model: Neural network model
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Computation device
        config: Configuration dictionary
    
    Returns:
        Dictionary with training history and results
    """
    # Setup training tracking
    train_metrics_history = []
    test_metrics_history = []
    best_iou = 0.0
    best_model_state = None
    lr_history = []
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Train one epoch
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        train_metrics_history.append(train_metrics)
        
        # Evaluate on test set
        test_metrics = evaluate(model, test_loader, device, criterion)
        test_metrics_history.append(test_metrics)
        
        # Update learning rate
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(test_metrics['iou'])
        else:
            scheduler.step()
        
        # Record current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train loss: {train_metrics['loss']:.4f}, "
              f"Test IoU: {test_metrics['iou']:.4f}, "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if test_metrics['iou'] > best_iou:
            best_iou = test_metrics['iou']
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print(f"Saved new best model with IoU: {best_iou:.4f}")
    
    # Training complete
    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best IoU: {best_iou:.4f} at epoch {best_epoch+1}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot([m['loss'] for m in train_metrics_history], label='Train')
    plt.plot([m['loss'] for m in test_metrics_history], label='Test')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot([m['iou'] for m in train_metrics_history], label='Train')
    plt.plot([m['iou'] for m in test_metrics_history], label='Test')
    plt.title('IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(lr_history)
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')
    plt.grid(True)
    
    plt.savefig(f"{config['save_dir']}/training_curves.png", dpi=200)
    plt.show()
    
    # Return results
    return {
        'train_metrics': train_metrics_history,
        'test_metrics': test_metrics_history,
        'lr_history': lr_history,
        'best_iou': best_iou,
        'best_epoch': best_epoch,
        'best_model_state': best_model_state,
        'training_time': time_elapsed
    }
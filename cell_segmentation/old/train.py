import time
import torch
from tqdm.auto import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import copy
from utils import calculate_metrics, EarlyStopping
import csv

def train_epoch(model, loader, optimizer, criterion, device, criterion_recon=None, config=None):
    """Train one epoch for both U-Net and W-Net"""
    model.train()
    epoch_loss = 0
    metrics = defaultdict(float)
    num_samples = 0
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        # Handle W-Net: if reconstruction loss is provided, assume two outputs from model
        if criterion_recon is not None:
            # Semi-supervised W-Net (segmentation loss + MSE)
            seg_output, recon_output = model(images)
            
            loss_seg = criterion(seg_output, masks)  # Supervised segmentation loss
            loss_recon = criterion_recon(recon_output, images)  # Reconstruction loss
            loss = loss_seg + loss_recon

        # U-Net: standard supervised segmentation with no reconstruction
        else:
            seg_output = model(images)
            loss = criterion(seg_output, masks)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            batch_metrics = calculate_metrics(torch.sigmoid(seg_output), masks)
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

def train_unsupervised_epoch(model, loader, optimizer, criterion_seg, criterion_recon, device, config):
    model.train()
    epoch_loss = 0
    num_samples = 0
    metrics = defaultdict(float)
    
    for images in loader:
        images = images.to(device)
        
        # For unsupervised W-Net, assume the model returns both segmentation and reconstruction outputs
        seg_output, recon_output = model(images)
        
        # Choose loss function based on the configuration
        if config.get('loss_fn') == 'softncut':
            loss_ncut = criterion_seg(images, seg_output)
            loss_recon = criterion_recon(recon_output, images)
            loss = config['ncut_weight'] * loss_ncut + config['recon_weight'] * loss_recon
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_size = images.size(0)
        epoch_loss += loss.item() * batch_size
        num_samples += batch_size
    
    
    epoch_loss /= num_samples
    for k in metrics:
        metrics[k] /= num_samples
    
    metrics['loss'] = epoch_loss
    return metrics

def evaluate(model, loader, device, criterion=None, criterion_recon=None, config=None):
    """Evaluate the model on the given data loader (supports U-Net and W-Net)"""
    model.eval()
    metrics = defaultdict(float)
    num_samples = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            # W-Net: segmentation + reconstruction
            if criterion_recon is not None:
                seg_output, recon_output = model(images)

                if config.get('loss_fn') == 'softncut':
                    # Unsupervised W-Net: NCut + MSE
                    loss_ncut = criterion(images, seg_output)
                    loss_recon = criterion_recon(recon_output, images)
                    loss = config['ncut_weight'] * loss_ncut + config['recon_weight'] * loss_recon
                    metrics['loss_ncut'] += loss_ncut.item() * images.size(0)
                    metrics['loss_recon'] += loss_recon.item() * images.size(0)
                else:
                    # Semi-supervised W-Net: segmentation + reconstruction
                    loss_seg = criterion(seg_output, masks)
                    loss_recon = criterion_recon(recon_output, images)
                    loss = loss_seg + loss_recon
                    metrics['loss_seg'] += loss_seg.item() * images.size(0)
                    metrics['loss_recon'] += loss_recon.item() * images.size(0)

            # U-Net: segmentation only
            else:
                seg_output = model(images)
                loss = criterion(seg_output, masks)
                metrics['loss_seg'] += loss.item() * images.size(0)

            # Compute IoU and other metrics
            preds = torch.sigmoid(seg_output)
            batch_metrics = calculate_metrics(preds, masks)

            for k, v in batch_metrics.items():
                metrics[k] += v * images.size(0)

            num_samples += images.size(0)

    # Normalize
    for k in metrics:
        metrics[k] /= num_samples

    return metrics

def train_model(train_loader, test_loader, model, criterion, optimizer, scheduler, num_epochs, device, config, criterion_recon=None):
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
    
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 7),
        min_delta=config.get('early_stopping_min_delta', 0.001)
    )

    metrics_csv_path = f"{config['save_dir']}/metrics_log.csv"
    with open(metrics_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "test_loss", "train_iou", "test_iou", "lr"])

    for epoch in range(num_epochs):
        # Train one epoch
        if config.get('loss_fn') == 'softncut':
            # Unsupervised training for W-Net
            train_metrics = train_unsupervised_epoch(model, train_loader, optimizer, criterion, criterion_recon, device, config)
        else:
            train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, criterion_recon, config=config)
        train_metrics_history.append(train_metrics)
        
        # Evaluate on test set
        test_metrics = evaluate(model, test_loader, device, criterion, criterion_recon=criterion_recon, config=config)
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
        
        # Log metrics to CSV
        with open(metrics_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                train_metrics['loss'],
                test_metrics.get('loss', test_metrics.get('loss_seg', 0.0)),
                train_metrics['iou'],
                test_metrics['iou'],
                current_lr
            ])
        
        # Save best model
        if test_metrics['iou'] > best_iou:
            best_iou = test_metrics['iou']
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print(f"Saved new best model with IoU: {best_iou:.4f}")

        # Early stopping check
        if early_stopping.step(test_metrics['iou']):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
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
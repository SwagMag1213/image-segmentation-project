import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupKFold
from datetime import datetime
import time
import copy
from collections import defaultdict
from torch.utils.data import DataLoader, Subset

# Import from our modular framework
from dataset import CellSegmentationDataset
from advanced_models import UNetWithBackbone
from losses import get_loss_function
from utils import get_device, calculate_metrics, EarlyStopping

def prepare_data_split_once(data_dir, image_type, test_size=0.2, img_size=(256, 256), seed=42):
    """
    Split data ONCE and return everything needed for all experiments
    This ensures all loss functions use exactly the same train/test split
    """
    print(f"Preparing data split (seed={seed})...")
    np.random.seed(seed)
    
    # Paths to augmented directories
    aug_dir = os.path.join(data_dir, f"augmented_{image_type}")
    aug_images_dir = os.path.join(aug_dir, "images")
    aug_masks_dir = os.path.join(aug_dir, "masks")
    
    if not os.path.exists(aug_dir):
        raise FileNotFoundError(f"Augmented directory {aug_dir} not found.")
    
    # Get all image files
    all_images = sorted(os.listdir(aug_images_dir))
    
    # Group images by their base original image
    image_groups = defaultdict(list)
    image_paths = []
    mask_paths = []
    
    for img in all_images:
        # Extract base name
        if "_orig.tif" in img:
            base_name = img.split('_orig.tif')[0]
        elif "_aug" in img:
            base_name = img.split('_aug')[0]
        else:
            continue
        
        # Check if corresponding mask exists
        mask_path = os.path.join(aug_masks_dir, img)
        if not os.path.exists(mask_path):
            continue
        
        # Add to paths
        img_path = os.path.join(aug_images_dir, img)
        image_paths.append(img_path)
        mask_paths.append(mask_path)
        
        # Group by base name
        idx = len(image_paths) - 1
        image_groups[base_name].append(idx)
    
    # Create dataset
    dataset = CellSegmentationDataset(image_paths, mask_paths, img_size=img_size)
    
    # Split base images into train/test
    base_names = list(image_groups.keys())
    train_bases, test_bases = train_test_split(
        base_names, test_size=test_size, random_state=seed
    )
    
    # Get indices for train and test sets
    train_indices = []
    test_indices = []
    
    for base in train_bases:
        train_indices.extend(image_groups[base])
    
    for base in test_bases:
        test_indices.extend(image_groups[base])
    
    print(f"Data split completed:")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Base images: {len(base_names)}")
    print(f"  Train: {len(train_indices)} images from {len(train_bases)} base images")
    print(f"  Test: {len(test_indices)} images from {len(test_bases)} base images")
    print(f"  Train bases: {sorted(train_bases)}")
    print(f"  Test bases: {sorted(test_bases)}")
    
    return {
        'dataset': dataset,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'image_groups': image_groups,
        'train_bases': train_bases,
        'test_bases': test_bases,
        'split_info': {
            'seed': seed,
            'test_size': test_size,
            'total_images': len(image_paths),
            'total_base_images': len(base_names),
            'train_images': len(train_indices),
            'test_images': len(test_indices)
        }
    }

def setup_cross_validation_folds(data_split, n_splits=5):
    """
    Set up cross-validation folds ONCE for the training set
    Returns fold indices that will be reused across all experiments
    """
    dataset = data_split['dataset']
    train_indices = data_split['train_indices']
    image_groups = data_split['image_groups']
    train_bases = data_split['train_bases']
    
    print(f"\nSetting up {n_splits}-fold cross-validation...")
    
    # Create groups array for GroupKFold (only for training indices)
    train_groups = []
    index_to_base = {}
    
    # Map each training index to its base image
    for base_name, indices in image_groups.items():
        if base_name in train_bases:
            for idx in indices:
                if idx in train_indices:
                    index_to_base[idx] = base_name
    
    # Create groups array in the same order as train_indices
    for idx in train_indices:
        train_groups.append(index_to_base[idx])
    
    # Set up GroupKFold
    group_kfold = GroupKFold(n_splits=n_splits)
    
    # Convert to numpy for sklearn
    X_train = np.array(train_indices)
    groups_array = np.array(train_groups)
    
    # Get unique groups and create numerical mapping
    unique_groups = sorted(list(set(train_groups)))
    group_to_idx = {group: idx for idx, group in enumerate(unique_groups)}
    group_indices = np.array([group_to_idx[group] for group in train_groups])
    
    # Generate all fold splits
    cv_folds = []
    for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(group_kfold.split(X_train, groups=group_indices)):
        fold_train_indices = X_train[train_fold_idx].tolist()
        fold_val_indices = X_train[val_fold_idx].tolist()
        
        cv_folds.append({
            'fold_num': fold_idx + 1,
            'train_indices': fold_train_indices,
            'val_indices': fold_val_indices
        })
        
        print(f"  Fold {fold_idx+1}: {len(fold_train_indices)} train, {len(fold_val_indices)} val")
    
    print(f"Cross-validation setup completed: {len(cv_folds)} folds")
    
    return cv_folds

def train_single_fold_single_loss(config, train_loader, val_loader, device, fold_num):
    """
    Train a single fold for a single loss function and return results
    """
    # Create model
    model = UNetWithBackbone(
        n_classes=1,
        backbone=config['backbone'],
        pretrained=config['pretrained'],
        use_attention=config['use_attention']
    ).to(device)
    
    # Create loss function and optimizer
    criterion = get_loss_function(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, threshold=0.01, min_lr=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 5),
        min_delta=config.get('early_stopping_min_delta', 0.001)
    )
    
    # Training tracking
    best_val_iou = 0.0
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_samples = 0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_samples += images.size(0)
            train_loss += loss.item() * images.size(0)
        
        train_loss /= train_samples
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_metrics = defaultdict(float)
        val_samples = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                batch_metrics = calculate_metrics(torch.sigmoid(outputs), masks)
                batch_size = images.size(0)
                
                for k, v in batch_metrics.items():
                    val_metrics[k] += v * batch_size
                val_samples += batch_size
                val_loss += loss.item() * batch_size
        
        # Normalize validation metrics
        val_loss /= val_samples
        for k in val_metrics:
            val_metrics[k] /= val_samples
        
        # Update learning rate
        scheduler.step(val_metrics['iou'])
        
        # Track best validation IoU
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
        
        # Early stopping
        if early_stopping.step(val_metrics['iou']):
            break
    
    # Return final validation metrics
    return dict(val_metrics)

def run_cross_validation_all_losses(configs, data_split, cv_folds):
    """
    Run cross-validation for ALL loss functions on each fold
    This gives us fair comparison across all loss functions
    """
    device = get_device()
    dataset = data_split['dataset']
    
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION: TESTING ALL LOSS FUNCTIONS ON EACH FOLD")
    print(f"{'='*60}")
    
    # Storage for all results
    # Structure: cv_results[loss_name][fold_num] = metrics
    cv_results = defaultdict(list)
    
    # Run each fold
    for fold_info in cv_folds:
        fold_num = fold_info['fold_num']
        fold_train_indices = fold_info['train_indices']
        fold_val_indices = fold_info['val_indices']
        
        print(f"\n--- Fold {fold_num}/{len(cv_folds)} ---")
        print(f"Training: {len(fold_train_indices)} images, Validation: {len(fold_val_indices)} images")
        
        # Create data loaders for this fold
        train_subset = Subset(dataset, fold_train_indices)
        val_subset = Subset(dataset, fold_val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=configs[0]['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=configs[0]['batch_size'], shuffle=False)
        
        # Test each loss function on this fold
        fold_results = {}
        for config in configs:
            loss_name = config['name']
            print(f"  Training {loss_name}...")
            
            start_time = time.time()
            val_metrics = train_single_fold_single_loss(config, train_loader, val_loader, device, fold_num)
            train_time = time.time() - start_time
            
            # Store results for this fold
            val_metrics['training_time'] = train_time
            fold_results[loss_name] = val_metrics
            cv_results[loss_name].append(val_metrics)
            
            print(f"    {loss_name}: IoU = {val_metrics['iou']:.4f}, F1 = {val_metrics['f1']:.4f} ({train_time:.1f}s)")
        
        # Print fold summary
        print(f"\n  Fold {fold_num} Summary (by IoU):")
        sorted_results = sorted(fold_results.items(), key=lambda x: x[1]['iou'], reverse=True)
        for i, (loss_name, metrics) in enumerate(sorted_results):
            print(f"    {i+1}. {loss_name:25}: {metrics['iou']:.4f}")
    
    # Calculate CV summary statistics for each loss function
    cv_summary = {}
    for loss_name in cv_results:
        metrics_summary = {}
        for metric in ['iou', 'f1', 'precision', 'recall']:
            values = [fold_metrics[metric] for fold_metrics in cv_results[loss_name]]
            metrics_summary[f'{metric}_mean'] = np.mean(values)
            metrics_summary[f'{metric}_std'] = np.std(values)
        
        cv_summary[loss_name] = metrics_summary
    
    # Print overall CV summary
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY (RANKED BY MEAN IoU)")
    print(f"{'='*60}")
    
    # Sort by mean IoU
    sorted_cv = sorted(cv_summary.items(), key=lambda x: x[1]['iou_mean'], reverse=True)
    for i, (loss_name, summary) in enumerate(sorted_cv):
        mean_iou = summary['iou_mean']
        std_iou = summary['iou_std']
        mean_f1 = summary['f1_mean']
        print(f"{i+1:2d}. {loss_name:25}: IoU = {mean_iou:.4f} ± {std_iou:.4f}, F1 = {mean_f1:.4f}")
    
    return cv_results, cv_summary

def retrain_on_full_training_set(config, data_split):
    """
    Retrain a single loss function on the full training set
    """
    device = get_device()
    dataset = data_split['dataset']
    train_indices = data_split['train_indices']
    
    print(f"  Retraining {config['name']} on full training set ({len(train_indices)} images)...")
    
    # Create full training loader
    train_subset = Subset(dataset, train_indices)
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    
    # Create model
    model = UNetWithBackbone(
        n_classes=1,
        backbone=config['backbone'],
        pretrained=config['pretrained'],
        use_attention=config['use_attention']
    ).to(device)
    
    # Create loss function and optimizer
    criterion = get_loss_function(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, threshold=0.01, min_lr=1e-6
    )
    
    # Training loop
    best_train_iou = 0.0
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        train_metrics = defaultdict(float)
        train_samples = 0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                batch_metrics = calculate_metrics(torch.sigmoid(outputs), masks)
                batch_size = images.size(0)
                
                for k, v in batch_metrics.items():
                    train_metrics[k] += v * batch_size
                train_samples += batch_size
                epoch_loss += loss.item() * batch_size
        
        # Normalize training metrics
        epoch_loss /= train_samples
        for k in train_metrics:
            train_metrics[k] /= train_samples
        
        # Update learning rate
        scheduler.step(train_metrics['iou'])
        
        if train_metrics['iou'] > best_train_iou:
            best_train_iou = train_metrics['iou']
    
    print(f"    Final training IoU: {best_train_iou:.4f}")
    
    return model, best_train_iou

def evaluate_generalization_error(configs, data_split):
    """
    Retrain each loss function on full training set and evaluate on test set
    This gives us the generalization error for each loss function
    """
    device = get_device()
    dataset = data_split['dataset']
    test_indices = data_split['test_indices']
    
    print(f"\n{'='*60}")
    print("FINAL TRAINING: RETRAIN ON FULL TRAINING SET & TEST GENERALIZATION")
    print(f"{'='*60}")
    
    # Create test loader
    test_subset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=configs[0]['batch_size'], shuffle=False)
    
    generalization_results = {}
    
    for config in configs:
        loss_name = config['name']
        print(f"\n--- {loss_name} ---")
        
        start_time = time.time()
        
        # Retrain on full training set
        model, final_train_iou = retrain_on_full_training_set(config, data_split)
        
        # Evaluate on test set
        model.eval()
        test_metrics = defaultdict(float)
        test_samples = 0
        
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                batch_metrics = calculate_metrics(torch.sigmoid(outputs), masks)
                batch_size = images.size(0)
                
                for k, v in batch_metrics.items():
                    test_metrics[k] += v * batch_size
                test_samples += batch_size
        
        # Normalize test metrics
        for k in test_metrics:
            test_metrics[k] /= test_samples
        
        total_time = time.time() - start_time
        
        generalization_results[loss_name] = {
            'final_train_iou': final_train_iou,
            'test_metrics': dict(test_metrics),
            'training_time': total_time
        }
        
        print(f"  Test IoU: {test_metrics['iou']:.4f}")
        print(f"  Test F1:  {test_metrics['f1']:.4f}")
        print(f"  Training time: {total_time:.1f}s")
    
    # Print generalization summary
    print(f"\n{'='*60}")
    print("GENERALIZATION ERROR SUMMARY (RANKED BY TEST IoU)")
    print(f"{'='*60}")
    
    sorted_gen = sorted(generalization_results.items(), 
                       key=lambda x: x[1]['test_metrics']['iou'], reverse=True)
    
    for i, (loss_name, results) in enumerate(sorted_gen):
        test_iou = results['test_metrics']['iou']
        test_f1 = results['test_metrics']['f1']
        train_iou = results['final_train_iou']
        overfitting = train_iou - test_iou
        print(f"{i+1:2d}. {loss_name:25}: Test IoU = {test_iou:.4f}, "
              f"F1 = {test_f1:.4f}, Overfitting = {overfitting:.4f}")
    
    return generalization_results

def plot_comprehensive_results(cv_summary, generalization_results, data_split, save_dir):
    """
    Create comprehensive comparison plots
    """
    plt.figure(figsize=(18, 12))
    
    # Prepare data
    loss_names = list(cv_summary.keys())
    cv_ious = [cv_summary[name]['iou_mean'] for name in loss_names]
    cv_stds = [cv_summary[name]['iou_std'] for name in loss_names]
    test_ious = [generalization_results[name]['test_metrics']['iou'] for name in loss_names]
    train_ious = [generalization_results[name]['final_train_iou'] for name in loss_names]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(loss_names)))
    
    # Plot 1: CV vs Test Performance
    plt.subplot(2, 4, 1)
    plt.scatter(cv_ious, test_ious, c=colors, s=100, alpha=0.7)
    for i, name in enumerate(loss_names):
        plt.annotate(name.replace('_loss', ''), (cv_ious[i], test_ious[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('CV Mean IoU')
    plt.ylabel('Test IoU (Generalization)')
    plt.title('Cross-Validation vs Generalization')
    plt.grid(True, alpha=0.3)
    
    # Add diagonal line
    min_val = min(min(cv_ious), min(test_ious))
    max_val = max(max(cv_ious), max(test_ious))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Plot 2: CV Performance Ranking
    plt.subplot(2, 4, 2)
    sorted_cv_idx = np.argsort(cv_ious)[::-1]
    sorted_names = [loss_names[i].replace('_loss', '') for i in sorted_cv_idx]
    sorted_cv_ious = [cv_ious[i] for i in sorted_cv_idx]
    sorted_cv_stds = [cv_stds[i] for i in sorted_cv_idx]
    
    bars = plt.bar(range(len(sorted_names)), sorted_cv_ious, 
                  yerr=sorted_cv_stds, capsize=5, color=[colors[i] for i in sorted_cv_idx], alpha=0.7)
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
    plt.ylabel('IoU')
    plt.title('CV Performance (with std)')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 3: Test Performance Ranking
    plt.subplot(2, 4, 3)
    sorted_test_idx = np.argsort(test_ious)[::-1]
    sorted_names_test = [loss_names[i].replace('_loss', '') for i in sorted_test_idx]
    sorted_test_ious = [test_ious[i] for i in sorted_test_idx]
    
    bars = plt.bar(range(len(sorted_names_test)), sorted_test_ious, 
                  color=[colors[i] for i in sorted_test_idx], alpha=0.7)
    plt.xticks(range(len(sorted_names_test)), sorted_names_test, rotation=45, ha='right')
    plt.ylabel('IoU')
    plt.title('Test Performance (Generalization)')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, iou in zip(bars, sorted_test_ious):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{iou:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Overfitting Analysis
    plt.subplot(2, 4, 4)
    overfitting = [train_ious[i] - test_ious[i] for i in range(len(loss_names))]
    bars = plt.bar(range(len(loss_names)), overfitting, color=colors, alpha=0.7)
    plt.xticks(range(len(loss_names)), [name.replace('_loss', '') for name in loss_names], 
              rotation=45, ha='right')
    plt.ylabel('Train IoU - Test IoU')
    plt.title('Overfitting Analysis')
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 5: Performance Comparison Table
    plt.subplot(2, 4, 5)
    plt.axis('off')
    
    # Sort by test IoU for table
    table_data = []
    for i in sorted_test_idx:
        name = loss_names[i]
        table_data.append([
            name.replace('_loss', ''),
            f"{cv_ious[i]:.3f}±{cv_stds[i]:.3f}",
            f"{test_ious[i]:.3f}",
            f"{overfitting[i]:.3f}"
        ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['Loss Function', 'CV IoU', 'Test IoU', 'Overfit'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    plt.title('Performance Summary\n(Ranked by Test IoU)', pad=20)
    
    # Plot 6: Data Split Info
    plt.subplot(2, 4, 6)
    split_info = data_split['split_info']
    train_images = split_info['train_images']
    test_images = split_info['test_images']
    
    plt.pie([train_images, test_images], 
            labels=[f'Train\n({train_images} images)', f'Test\n({test_images} images)'],
            autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
    plt.title('Train/Test Split')
    
    # Plot 7: Cross-Validation vs Test Correlation
    plt.subplot(2, 4, 7)
    plt.scatter(cv_ious, test_ious, c=colors, s=100, alpha=0.7)
    
    # Calculate correlation
    correlation = np.corrcoef(cv_ious, test_ious)[0, 1]
    
    # Fit line
    z = np.polyfit(cv_ious, test_ious, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(cv_ious), max(cv_ious), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8)
    
    plt.xlabel('CV Mean IoU')
    plt.ylabel('Test IoU')
    plt.title(f'CV-Test Correlation\nr = {correlation:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Method Ranking Comparison
    plt.subplot(2, 4, 8)
    cv_ranks = [sorted_cv_idx.tolist().index(i) + 1 for i in range(len(loss_names))]
    test_ranks = [sorted_test_idx.tolist().index(i) + 1 for i in range(len(loss_names))]
    
    for i, name in enumerate(loss_names):
        plt.plot([1, 2], [cv_ranks[i], test_ranks[i]], 'o-', color=colors[i], 
                alpha=0.7, label=name.replace('_loss', ''))
    
    plt.xticks([1, 2], ['CV Rank', 'Test Rank'])
    plt.ylabel('Rank (1 = best)')
    plt.title('Ranking Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # Invert so rank 1 is at top
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comprehensive_loss_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def get_loss_configurations():
    """Define loss function configurations to test"""
    base_config = {
        'backbone': 'resnet34',
        'use_attention': False,
        'batch_size': 2,
        'img_size': (128, 128),
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'pretrained': True,
        'seed': 42,
        'image_type': 'W',
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 0.001,
    }
    
    return [
        {**base_config, 'name': 'dice_loss', 'loss_fn': 'dice'},
        {**base_config, 'name': 'bce_loss', 'loss_fn': 'bce'},
        {**base_config, 'name': 'bce_dice_combo_loss', 'loss_fn': 'combo', 'loss_alpha': 0.5},
        {**base_config, 'name': 'focal_loss', 'loss_fn': 'focal', 'focal_alpha': 0.25, 'focal_gamma': 2.0},
        {**base_config, 'name': 'focal_bce_dice_combo_loss', 'loss_fn': 'triple_combo'},
        {**base_config, 'name': 'tversky_balanced', 'loss_fn': 'tversky_balanced'},
        {**base_config, 'name': 'boundary_loss', 'loss_fn': 'boundary'},
    ]

def main():
    """Main experiment runner with improved CV design"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"experiments/improved_loss_comparison_{timestamp}"

    os.makedirs(save_dir, exist_ok=True)
    data_dir = "data/manual_labels"
    
    print("="*80)
    print("IMPROVED LOSS FUNCTION COMPARISON EXPERIMENT")
    print("="*80)
    print("Methodology:")
    print("1. Split data ONCE into train (80%) / test (20%) by base images")
    print("2. Set up cross-validation folds ONCE for training set")
    print("3. For EACH fold: test ALL loss functions")
    print("4. Average CV performance across folds for each loss function")
    print("5. Retrain each loss function on FULL training set")
    print("6. Test each final model on held-out test set (generalization error)")
    print("="*80)
    
    # Get configurations
    configs = get_loss_configurations()
    seed = configs[0]['seed']
    image_type = configs[0]['image_type']
    img_size = configs[0]['img_size']
    
    # STEP 1: Split data once
    data_split = prepare_data_split_once(
        data_dir=data_dir,
        image_type=image_type,
        test_size=0.2,
        img_size=img_size,
        seed=seed
    )
    
    # STEP 2: Set up CV folds once
    cv_folds = setup_cross_validation_folds(data_split, n_splits=5)
    
    # Save experimental setup
    torch.save({
        'data_split': data_split,
        'cv_folds': cv_folds,
        'configs': configs,
        'timestamp': timestamp
    }, f"{save_dir}/experimental_setup.pth")
    
    # STEP 3: Run cross-validation (all losses on each fold)
    cv_results, cv_summary = run_cross_validation_all_losses(configs, data_split, cv_folds)
    
    # STEP 4: Retrain and evaluate generalization error
    generalization_results = evaluate_generalization_error(configs, data_split)
    
    # STEP 5: Create comprehensive plots
    print(f"\n{'='*60}")
    print("GENERATING COMPREHENSIVE COMPARISON PLOTS")
    print(f"{'='*60}")
    
    plot_comprehensive_results(cv_summary, generalization_results, data_split, save_dir)
    
    # Save all results
    torch.save({
        'cv_results': cv_results,
        'cv_summary': cv_summary,
        'generalization_results': generalization_results,
        'configs': configs
    }, f"{save_dir}/all_results.pth")
    
    # STEP 6: Final summary
    print(f"\n{'='*80}")
    print("FINAL EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    print("\nBest by Cross-Validation:")
    best_cv = max(cv_summary.items(), key=lambda x: x[1]['iou_mean'])
    print(f"  {best_cv[0]}: {best_cv[1]['iou_mean']:.4f} ± {best_cv[1]['iou_std']:.4f}")
    
    print("\nBest by Generalization (Test Set):")
    best_test = max(generalization_results.items(), key=lambda x: x[1]['test_metrics']['iou'])
    print(f"  {best_test[0]}: {best_test[1]['test_metrics']['iou']:.4f}")
    
    # Check if CV and test winners are the same
    if best_cv[0] == best_test[0]:
        print(f"\n✅ CONSISTENT WINNER: {best_cv[0]}")
        print("   Cross-validation successfully identified the best loss function!")
    else:
        print(f"\n⚠️  DIFFERENT WINNERS:")
        print(f"   CV Best: {best_cv[0]}")
        print(f"   Test Best: {best_test[0]}")
        print("   Consider using CV winner for model selection.")
    
    # Calculate CV-Test correlation
    cv_ious = [cv_summary[name]['iou_mean'] for name in cv_summary.keys()]
    test_ious = [generalization_results[name]['test_metrics']['iou'] for name in cv_summary.keys()]
    correlation = np.corrcoef(cv_ious, test_ious)[0, 1]
    print(f"\nCV-Test Correlation: {correlation:.3f}")
    if correlation > 0.8:
        print("   Strong correlation - CV is reliable for model selection")
    elif correlation > 0.5:
        print("   Moderate correlation - CV provides reasonable guidance")
    else:
        print("   Weak correlation - CV may not be reliable for this dataset")
    
    print(f"\nAll results saved to: {save_dir}")

if __name__ == "__main__":
    main()
"""
Comprehensive testing for optimal number of CV folds and augmentation amount
"""

import os
import torch
import numpy as np
import albumentations as A
import cv2
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime
import shutil

# Import your existing modules
from dataset import CellSegmentationDataset
from advanced_models import UNetWithBackbone
from losses import get_loss_function
from utils import get_device, calculate_metrics, EarlyStopping
from train import train_epoch, evaluate


class FoldAugmentationTester:
    """Test different numbers of CV folds and augmentation amounts."""
    
    def __init__(self, 
                 base_config: Dict,
                 data_dir: str,
                 selected_augmentations: List[str],
                 fold_options: List[int] = [3, 5, 7, 10],
                 augmentation_amounts: List[int] = [1, 2, 3, 5, 10]):
        """
        Args:
            base_config: Your existing config dictionary
            data_dir: Path to your data directory
            selected_augmentations: List of augmentation names from forward selection
            fold_options: Different numbers of CV folds to test
            augmentation_amounts: Different augmentation multipliers to test
        """
        self.base_config = base_config
        self.data_dir = data_dir
        self.selected_augmentations = selected_augmentations
        self.fold_options = fold_options
        self.augmentation_amounts = augmentation_amounts
        
        # Load dataset
        self._load_dataset_paths()
        
        # Create augmentation pipeline with realistic probabilities
        self.aug_pipeline = self._create_production_pipeline()
        
        # Results storage
        self.results = {
            'fold_results': {},
            'augmentation_results': {},
            'detailed_metrics': []
        }
        
        # Keep track of temp directories to clean up
        self.temp_dirs = []
        
    def _load_dataset_paths(self):
        """Load original dataset paths."""
        images_dir = os.path.join(self.data_dir, "Labelled_images")
        masks_dir = os.path.join(self.data_dir, "GT_masks")
        
        all_masks = sorted(os.listdir(masks_dir))
        all_images = sorted(os.listdir(images_dir))
        
        self.image_paths = []
        self.mask_paths = []
        self.base_names = []
        
        for mask_file in all_masks:
            if not mask_file.endswith('GT.tif'):
                continue
            
            parts = mask_file.split('_')
            img_type = parts[3][1]
            
            if img_type != self.base_config['image_type']:
                continue
            
            original_file = mask_file[:-7] + '.tif'
            
            if original_file in all_images:
                img_path = os.path.join(images_dir, original_file)
                mask_path = os.path.join(masks_dir, mask_file)
                
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
                
                base_name = mask_file[:-7]
                self.base_names.append(base_name)
        
        self.unique_bases = sorted(list(set(self.base_names)))
        print(f"Loaded {len(self.image_paths)} images from {len(self.unique_bases)} unique base images")
        
    def _create_production_pipeline(self) -> A.Compose:
        """Create augmentation pipeline with production probabilities."""
        # Define the selected augmentations with realistic probabilities
        augmentation_configs = {
            'random_rotate_90': {
                'transform': A.RandomRotate90(),
                'prob': 0.5  # 50% chance for discrete rotations
            },
            'affine': {
                'transform': A.Affine(
                    scale=(0.95, 1.05),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-15, 15),
                    shear=(-5, 5),
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT
                ),
                'prob': 0.3  # 30% for complex geometric
            },
            'vertical_flip': {
                'transform': A.VerticalFlip(),
                'prob': 0.5  # 50% for flips
            },
            'advanced_blur': {
                'transform': A.AdvancedBlur(
                    blur_limit=(3, 7),
                    sigmaX_limit=(0.2, 1.0),
                    sigmaY_limit=(0.2, 1.0),
                    rotate_limit=90,
                    beta_limit=(0.5, 8.0),
                    noise_limit=(0.9, 1.1)
                ),
                'prob': 0.3  # 30% for blur
            }
        }
        
        transforms = []
        for aug_name in self.selected_augmentations:
            if aug_name in augmentation_configs:
                config = augmentation_configs[aug_name]
                transform = config['transform']
                transform.p = config['prob']
                transforms.append(transform)
        
        return A.Compose(transforms)
    
    def _create_augmented_dataset(self, image_paths: List[str], mask_paths: List[str], 
                                  num_augmentations: int) -> Tuple[List[str], List[str]]:
        """Create augmented dataset with specified number of augmentations per image."""
        if num_augmentations == 0:
            return image_paths, mask_paths
        
        expanded_images = image_paths.copy()
        expanded_masks = mask_paths.copy()
        
        # Create temp directory
        temp_dir = f"temp_fold_aug_test_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        os.makedirs(temp_dir, exist_ok=True)
        self.temp_dirs.append(temp_dir)  # Track for cleanup
        
        for img_path, mask_path in zip(image_paths, mask_paths):
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8) * 255
            
            for aug_idx in range(num_augmentations):
                # Apply augmentation pipeline
                augmented = self.aug_pipeline(image=image, mask=mask)
                
                # Save augmented version
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                temp_img_path = os.path.join(temp_dir, f"{base_name}_aug_{aug_idx}.tif")
                temp_mask_path = os.path.join(temp_dir, f"{base_name}_mask_aug_{aug_idx}.tif")
                
                cv2.imwrite(temp_img_path, augmented['image'])
                cv2.imwrite(temp_mask_path, augmented['mask'])
                
                expanded_images.append(temp_img_path)
                expanded_masks.append(temp_mask_path)
        
        return expanded_images, expanded_masks
    
    def _train_and_evaluate_model(self, train_imgs: List[str], train_masks: List[str],
                                  val_imgs: List[str], val_masks: List[str],
                                  num_augmentations: int, max_epochs: int = 30) -> Dict:
        """Train model and return detailed metrics."""
        # Create augmented training data
        train_imgs_aug, train_masks_aug = self._create_augmented_dataset(
            train_imgs, train_masks, num_augmentations
        )
        
        # Create datasets
        train_dataset = CellSegmentationDataset(
            train_imgs_aug, train_masks_aug, 
            img_size=self.base_config['img_size']
        )
        val_dataset = CellSegmentationDataset(
            val_imgs, val_masks,
            img_size=self.base_config['img_size']
        )
        
        # Create data loaders
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.base_config['batch_size'], 
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.base_config['batch_size'], 
            shuffle=False,
            num_workers=2
        )
        
        # Create model
        model = UNetWithBackbone(
            n_classes=1,
            backbone=self.base_config['backbone'],
            pretrained=self.base_config['pretrained'],
            use_attention=self.base_config['use_attention']
        )
        
        device = get_device()
        model = model.to(device)
        
        # Create loss and optimizer
        criterion = get_loss_function(self.base_config)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.base_config['learning_rate'],
            weight_decay=self.base_config.get('weight_decay', 1e-5)
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=7, min_delta=0.001)
        
        # Training metrics storage
        train_history = {'loss': [], 'iou': []}
        val_history = {'loss': [], 'iou': []}
        best_iou = 0.0
        best_epoch = 0
        
        # Training loop
        for epoch in range(max_epochs):
            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
            train_history['loss'].append(train_metrics['loss'])
            train_history['iou'].append(train_metrics['iou'])
            
            # Validate
            val_metrics = evaluate(model, val_loader, device, criterion)
            val_history['loss'].append(val_metrics['loss'])
            val_history['iou'].append(val_metrics['iou'])
            
            # Update scheduler
            scheduler.step(val_metrics['iou'])
            
            # Track best
            if val_metrics['iou'] > best_iou:
                best_iou = val_metrics['iou']
                best_epoch = epoch
            
            # Early stopping
            if early_stopping.step(val_metrics['iou']):
                break
        
        return {
            'best_iou': best_iou,
            'best_epoch': best_epoch,
            'final_epoch': epoch,
            'train_history': train_history,
            'val_history': val_history,
            'final_metrics': val_metrics
        }
    
    def test_fold_variations(self, num_augmentations: int = 3):
        """Test different numbers of CV folds with fixed augmentation amount."""
        print(f"\n{'='*60}")
        print(f"TESTING FOLD VARIATIONS (with {num_augmentations} augmentations per image)")
        print(f"{'='*60}")
        
        for n_folds in self.fold_options:
            print(f"\nTesting {n_folds}-fold CV:")
            
            # Check if we have enough unique base images
            if n_folds > len(self.unique_bases):
                print(f"  ⚠️  Skipping: Only {len(self.unique_bases)} unique base images available")
                continue
            
            # Perform CV
            group_kfold = GroupKFold(n_splits=n_folds)
            base_to_idx = {base: idx for idx, base in enumerate(self.unique_bases)}
            group_indices = np.array([base_to_idx[base] for base in self.base_names])
            
            fold_scores = []
            X = np.arange(len(self.image_paths))
            
            for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, groups=group_indices)):
                print(f"  Fold {fold + 1}/{n_folds}:", end=" ")
                
                # Split data
                train_imgs = [self.image_paths[i] for i in train_idx]
                train_masks = [self.mask_paths[i] for i in train_idx]
                val_imgs = [self.image_paths[i] for i in val_idx]
                val_masks = [self.mask_paths[i] for i in val_idx]
                
                # Count unique base images in train/val
                train_bases = set([self.base_names[i] for i in train_idx])
                val_bases = set([self.base_names[i] for i in val_idx])
                print(f"Train: {len(train_bases)} base images, Val: {len(val_bases)} base images", end=" ")
                
                # Train and evaluate
                metrics = self._train_and_evaluate_model(
                    train_imgs, train_masks, val_imgs, val_masks, num_augmentations
                )
                
                fold_scores.append(metrics['best_iou'])
                print(f"IoU: {metrics['best_iou']:.4f}")
                
                # Store detailed metrics
                self.results['detailed_metrics'].append({
                    'test_type': 'fold_variation',
                    'n_folds': n_folds,
                    'fold': fold + 1,
                    'num_augmentations': num_augmentations,
                    'train_base_images': len(train_bases),
                    'val_base_images': len(val_bases),
                    'best_iou': metrics['best_iou'],
                    'best_epoch': metrics['best_epoch'],
                    'final_epoch': metrics['final_epoch']
                })
            
            # Store results
            mean_iou = np.mean(fold_scores)
            std_iou = np.std(fold_scores)
            self.results['fold_results'][n_folds] = {
                'mean_iou': mean_iou,
                'std_iou': std_iou,
                'fold_scores': fold_scores
            }
            
            print(f"  Overall: {mean_iou:.4f} ± {std_iou:.4f}")
    
    def test_augmentation_amounts(self, n_folds: int = 5):
        """Test different augmentation amounts with fixed number of folds."""
        print(f"\n{'='*60}")
        print(f"TESTING AUGMENTATION AMOUNTS (with {n_folds}-fold CV)")
        print(f"{'='*60}")
        
        for num_aug in self.augmentation_amounts:
            print(f"\nTesting {num_aug} augmentation(s) per image:")
            
            # Perform CV
            group_kfold = GroupKFold(n_splits=n_folds)
            base_to_idx = {base: idx for idx, base in enumerate(self.unique_bases)}
            group_indices = np.array([base_to_idx[base] for base in self.base_names])
            
            fold_scores = []
            X = np.arange(len(self.image_paths))
            
            for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, groups=group_indices)):
                print(f"  Fold {fold + 1}/{n_folds}:", end=" ")
                
                # Split data
                train_imgs = [self.image_paths[i] for i in train_idx]
                train_masks = [self.mask_paths[i] for i in train_idx]
                val_imgs = [self.image_paths[i] for i in val_idx]
                val_masks = [self.mask_paths[i] for i in val_idx]
                
                # Calculate total training samples
                total_train_samples = len(train_imgs) * (1 + num_aug)
                print(f"Training samples: {total_train_samples}", end=" ")
                
                # Train and evaluate
                metrics = self._train_and_evaluate_model(
                    train_imgs, train_masks, val_imgs, val_masks, num_aug
                )
                
                fold_scores.append(metrics['best_iou'])
                print(f"IoU: {metrics['best_iou']:.4f}")
                
                # Store detailed metrics
                self.results['detailed_metrics'].append({
                    'test_type': 'augmentation_amount',
                    'n_folds': n_folds,
                    'fold': fold + 1,
                    'num_augmentations': num_aug,
                    'total_train_samples': total_train_samples,
                    'best_iou': metrics['best_iou'],
                    'best_epoch': metrics['best_epoch'],
                    'final_epoch': metrics['final_epoch']
                })
            
            # Store results
            mean_iou = np.mean(fold_scores)
            std_iou = np.std(fold_scores)
            self.results['augmentation_results'][num_aug] = {
                'mean_iou': mean_iou,
                'std_iou': std_iou,
                'fold_scores': fold_scores
            }
            
            print(f"  Overall: {mean_iou:.4f} ± {std_iou:.4f}")
    
    def run_all_tests(self):
        """Run both fold and augmentation tests independently."""
        print("\n" + "="*60)
        print("RUNNING INDEPENDENT TESTS")
        print("="*60)
        
        # Test 1: Fold variations with fixed augmentation amount (3)
        print("\nTEST 1: FOLD VARIATIONS")
        print("-"*40)
        self.test_fold_variations(num_augmentations=3)
        
        # Test 2: Augmentation amounts with fixed folds (5)
        print("\nTEST 2: AUGMENTATION AMOUNTS")
        print("-"*40)
        self.test_augmentation_amounts(n_folds=5)
        
        # Clean up all temp directories
        self._cleanup_all_temp_dirs()
        
        # Print summary
        self._print_summary()
    
    def _cleanup_all_temp_dirs(self):
        """Clean up all temporary directories created during testing."""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        self.temp_dirs = []
    
    def _print_summary(self):
        """Print summary of all test results."""
        print("\n" + "="*60)
        print("SUMMARY OF RESULTS")
        print("="*60)
        
        # Fold results summary
        if self.results['fold_results']:
            print("\nFOLD VARIATION RESULTS:")
            print("-"*40)
            fold_data = []
            for n_folds, results in sorted(self.results['fold_results'].items()):
                fold_data.append({
                    'Folds': n_folds,
                    'Mean IoU': f"{results['mean_iou']:.4f}",
                    'Std IoU': f"{results['std_iou']:.4f}",
                    'Train/Val Split': f"{int((1 - 1/n_folds) * len(self.unique_bases))}/{int(1/n_folds * len(self.unique_bases))}"
                })
            
            # Find best fold number
            best_fold = max(self.results['fold_results'].items(), 
                          key=lambda x: x[1]['mean_iou'])
            
            for data in fold_data:
                marker = "→" if data['Folds'] == best_fold[0] else " "
                print(f"{marker} {data['Folds']} folds: {data['Mean IoU']} ± {data['Std IoU'].replace('±', '')} "
                      f"(~{data['Train/Val Split']} base images)")
            
            print(f"\nBest fold configuration: {best_fold[0]} folds with IoU = {best_fold[1]['mean_iou']:.4f}")
        
        # Augmentation results summary
        if self.results['augmentation_results']:
            print("\nAUGMENTATION AMOUNT RESULTS:")
            print("-"*40)
            aug_data = []
            for num_aug, results in sorted(self.results['augmentation_results'].items()):
                total_multiplier = 1 + num_aug
                aug_data.append({
                    'Augmentations': num_aug,
                    'Mean IoU': f"{results['mean_iou']:.4f}",
                    'Std IoU': f"{results['std_iou']:.4f}",
                    'Data Multiplier': f"{total_multiplier}x"
                })
            
            # Find best augmentation amount
            best_aug = max(self.results['augmentation_results'].items(), 
                         key=lambda x: x[1]['mean_iou'])
            
            for data in aug_data:
                marker = "→" if data['Augmentations'] == best_aug[0] else " "
                print(f"{marker} {data['Augmentations']} aug/image: {data['Mean IoU']} ± {data['Std IoU'].replace('±', '')} "
                      f"({data['Data Multiplier']} data)")
            
            print(f"\nBest augmentation amount: {best_aug[0]} per image with IoU = {best_aug[1]['mean_iou']:.4f}")
        
        print("\n" + "="*60)
    
    def plot_results(self, save_dir: str = 'fold_aug_test_results'):
        """Create visualization of results - simplified for independent tests."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Fold variation plot
        if self.results['fold_results']:
            folds = sorted(self.results['fold_results'].keys())
            means = [self.results['fold_results'][f]['mean_iou'] for f in folds]
            stds = [self.results['fold_results'][f]['std_iou'] for f in folds]
            
            ax1.errorbar(folds, means, yerr=stds, marker='o', linewidth=2, markersize=10, 
                        capsize=5, capthick=2, color='#1f77b4')
            ax1.set_xlabel('Number of CV Folds', fontsize=12)
            ax1.set_ylabel('IoU Score', fontsize=12)
            ax1.set_title('Effect of CV Folds on Model Performance', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for f, m, s in zip(folds, means, stds):
                ax1.annotate(f'{m:.3f}', (f, m + s + 0.003), ha='center', fontsize=10)
            
            # Highlight best
            best_fold_idx = np.argmax(means)
            ax1.scatter(folds[best_fold_idx], means[best_fold_idx], 
                       color='red', s=200, zorder=5, marker='*')
        
        # 2. Augmentation amount plot
        if self.results['augmentation_results']:
            augs = sorted(self.results['augmentation_results'].keys())
            means = [self.results['augmentation_results'][a]['mean_iou'] for a in augs]
            stds = [self.results['augmentation_results'][a]['std_iou'] for a in augs]
            
            ax2.errorbar(augs, means, yerr=stds, marker='s', linewidth=2, markersize=10, 
                        capsize=5, capthick=2, color='#2ca02c')
            ax2.set_xlabel('Number of Augmentations per Image', fontsize=12)
            ax2.set_ylabel('IoU Score', fontsize=12)
            ax2.set_title('Effect of Augmentation Amount on Model Performance', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for a, m, s in zip(augs, means, stds):
                ax2.annotate(f'{m:.3f}', (a, m + s + 0.003), ha='center', fontsize=10)
            
            # Highlight best
            best_aug_idx = np.argmax(means)
            ax2.scatter(augs[best_aug_idx], means[best_aug_idx], 
                       color='red', s=200, zorder=5, marker='*')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'fold_augmentation_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Training samples vs performance plot
        if self.results['detailed_metrics']:
            plt.figure(figsize=(10, 6))
            df = pd.DataFrame(self.results['detailed_metrics'])
            
            if 'augmentation_amount' in df['test_type'].values:
                aug_df = df[df['test_type'] == 'augmentation_amount']
                
                # Plot each fold as a separate line
                for fold in aug_df['fold'].unique():
                    fold_data = aug_df[aug_df['fold'] == fold]
                    plt.plot(fold_data['total_train_samples'], fold_data['best_iou'], 
                            'o-', alpha=0.5, label=f'Fold {fold}')
                
                # Add average line
                grouped = aug_df.groupby('num_augmentations').agg({
                    'total_train_samples': 'mean',
                    'best_iou': 'mean'
                }).reset_index()
                
                plt.plot(grouped['total_train_samples'], grouped['best_iou'], 
                        'o-', linewidth=3, markersize=10, color='black', label='Average')
                
                # Add augmentation labels
                for _, row in grouped.iterrows():
                    plt.annotate(f'{row["num_augmentations"]} aug', 
                               (row['total_train_samples'], row['best_iou']),
                               xytext=(5, -15), textcoords='offset points', fontsize=10)
                
                plt.xlabel('Total Training Samples', fontsize=12)
                plt.ylabel('IoU Score', fontsize=12)
                plt.title('Training Efficiency: Dataset Size vs Performance', fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'training_efficiency.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"\nPlots saved to {save_dir}/")
    
    def save_results(self, save_dir: str = 'fold_aug_test_results'):
        """Save all results to JSON."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save main results
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary
        summary = {
            'dataset': {
                'total_images': len(self.image_paths),
                'unique_base_images': len(self.unique_bases),
                'image_type': self.base_config['image_type']
            },
            'selected_augmentations': self.selected_augmentations,
            'test_configuration': {
                'fold_options': self.fold_options,
                'augmentation_amounts': self.augmentation_amounts
            }
        }
        
        # Add best results from independent tests
        if self.results['fold_results']:
            best_fold = max(self.results['fold_results'].items(), 
                          key=lambda x: x[1]['mean_iou'])
            summary['best_fold_configuration'] = {
                'n_folds': best_fold[0],
                'mean_iou': best_fold[1]['mean_iou'],
                'std_iou': best_fold[1]['std_iou']
            }
        
        if self.results['augmentation_results']:
            best_aug = max(self.results['augmentation_results'].items(), 
                         key=lambda x: x[1]['mean_iou'])
            summary['best_augmentation_amount'] = {
                'augmentations_per_image': best_aug[0],
                'mean_iou': best_aug[1]['mean_iou'],
                'std_iou': best_aug[1]['std_iou']
            }
        
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {save_dir}/")


def run_comprehensive_fold_augmentation_test():
    """Main function to run all tests."""
    
    # Configuration
    config = {
        'name': 'fold_augmentation_test',
        'model_type': 'unet',
        'image_type': 'W',  # or 'B'
        'backbone': 'resnet34',
        'use_attention': True,
        'batch_size': 4,
        'img_size': (128, 128),
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'pretrained': True,
        'seed': 42,
        'loss_fn': 'focal',
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
    }
    
    # Selected augmentations from your forward selection
    selected_augmentations = ['random_rotate_90', 'affine', 'vertical_flip', 'advanced_blur']
    
    # Data directory
    data_dir = 'manual_labels'
    
    print("="*60)
    print("FOLD AND AUGMENTATION AMOUNT TESTING")
    print("="*60)
    print(f"Model: {config['backbone']} (pretrained={config['pretrained']})")
    print(f"Image type: {config['image_type']}")
    print(f"Selected augmentations: {', '.join(selected_augmentations)}")
    print("="*60)
    
    # Initialize tester
    tester = FoldAugmentationTester(
        base_config=config,
        data_dir=data_dir,
        selected_augmentations=selected_augmentations,
        fold_options=[2, 3, 4, 5],  # Test these fold numbers
        augmentation_amounts=[1, 2, 3, 5, 10]  # Test these augmentation amounts (0 = baseline)
    )
    
    # Run all tests
    tester.run_all_tests()
    
    # Save results
    tester.save_results()
    
    # Create visualizations
    tester.plot_results()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print("="*60)
    
    return tester.results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run the comprehensive test
    results = run_comprehensive_fold_augmentation_test()
"""
Simplified Cross-Validation Module
Focused on CV only - test sets handled separately when needed
"""

import os
import torch
import numpy as np
import time
from typing import List, Dict, Callable, Tuple
from collections import defaultdict
from sklearn.model_selection import KFold

from dataset import load_original_data
from train import train_model
from utils import get_device
from losses import get_loss_function


class CrossValidator:
    """Simple cross-validation for model evaluation."""
    
    def __init__(self, data_dir: str = "manual_labels", image_type: str = 'W',
                 n_splits: int = 5, random_state: int = 42, 
                 augmentations_per_image: int = 3, verbose: bool = True):
        """
        Initialize CV with data loading only - no train/test split.
        
        Args:
            data_dir: Directory containing data
            image_type: 'W' or 'B' 
            n_splits: Number of CV folds
            random_state: Random seed
            augmentations_per_image: Augmentations per original image
            verbose: Print progress
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.augmentations_per_image = augmentations_per_image
        self.verbose = verbose
        
        # Load all data (no train/test split)
        self.data = load_original_data(data_dir, image_type)
        self.image_paths = self.data['image_paths']
        self.mask_paths = self.data['mask_paths']
        
        if verbose:
            print(f"Loaded {len(self.image_paths)} {image_type} images for CV")
    
    def create_cv_folds(self, indices: List[int] = None) -> List[Tuple[List[int], List[int]]]:
        """
        Create cross-validation folds.
        
        Args:
            indices: Specific indices to use for CV (if None, use all data)
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        if indices is None:
            indices = list(range(len(self.image_paths)))
        
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        cv_folds = []
        for train_fold_idx, val_fold_idx in kfold.split(indices):
            # Convert to actual indices
            fold_train_indices = [indices[i] for i in train_fold_idx]
            fold_val_indices = [indices[i] for i in val_fold_idx]
            cv_folds.append((fold_train_indices, fold_val_indices))
        
        if self.verbose:
            print(f"Created {self.n_splits} CV folds from {len(indices)} samples")
        
        return cv_folds
    
    def train_single_model(self, model_class: Callable, config: Dict, 
                          train_images: List[str], train_masks: List[str],
                          val_images: List[str], val_masks: List[str]) -> Dict:
        """Train a single model using the train module."""
        device = get_device()
        
        # Create model
        model = model_class(
            n_classes=1,
            backbone=config['backbone'],
            pretrained=config['pretrained'],
            use_attention=config['use_attention']
        ).to(device)
        
        # Setup training components
        criterion = get_loss_function(config)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
        )
        
        # Make config non-verbose for CV
        cv_config = config.copy()
        cv_config['verbose'] = False
        cv_config['save_plots'] = False
        
        # Train using the train module
        results = train_model(
            model=model,
            train_images=train_images,
            train_masks=train_masks,
            val_images=val_images,
            val_masks=val_masks,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config['num_epochs'],
            device=device,
            config=cv_config,
            augmentations_per_image=self.augmentations_per_image,
            save_plots=False
        )
        
        return results['final_val_metrics']
    
    def cross_validate_single_model(self, model_class: Callable, config: Dict, 
                                   indices: List[int] = None) -> Dict:
        """
        Perform cross-validation on a single model.
        
        Args:
            model_class: Model class to instantiate
            config: Configuration dictionary
            indices: Specific indices to use (if None, use all data)
            
        Returns:
            Dictionary with CV results
        """
        if self.verbose:
            print(f"\nCross-validating {config.get('name', 'Model')}...")
        
        # Create CV folds
        cv_folds = self.create_cv_folds(indices)
        
        # Store results
        fold_results = []
        all_metrics = defaultdict(list)
        
        # Run each fold
        for fold_idx, (fold_train_indices, fold_val_indices) in enumerate(cv_folds):
            if self.verbose:
                print(f"  Fold {fold_idx + 1}/{self.n_splits}:", end=" ")
            
            # Get file paths for this fold
            fold_train_images = [self.image_paths[i] for i in fold_train_indices]
            fold_train_masks = [self.mask_paths[i] for i in fold_train_indices]
            fold_val_images = [self.image_paths[i] for i in fold_val_indices]
            fold_val_masks = [self.mask_paths[i] for i in fold_val_indices]
            
            # Train model
            start_time = time.time()
            val_metrics = self.train_single_model(
                model_class, config, fold_train_images, fold_train_masks,
                fold_val_images, fold_val_masks
            )
            training_time = time.time() - start_time
            
            # Store results
            val_metrics['training_time'] = training_time
            fold_results.append(val_metrics)
            
            for metric, value in val_metrics.items():
                if metric != 'training_time':
                    all_metrics[metric].append(value)
            
            if self.verbose:
                print(f"IoU: {val_metrics['iou']:.4f} ({training_time:.1f}s)")
        
        # Calculate summary statistics
        cv_summary = {}
        for metric, values in all_metrics.items():
            cv_summary[f'{metric}_mean'] = np.mean(values)
            cv_summary[f'{metric}_std'] = np.std(values)
        
        if self.verbose:
            mean_iou = cv_summary['iou_mean']
            std_iou = cv_summary['iou_std']
            print(f"  Overall: {mean_iou:.4f} ± {std_iou:.4f}")
        
        return {
            'config': config,
            'fold_results': fold_results,
            'cv_summary': cv_summary
        }
    
    def compare_multiple_models(self, model_configs: List[Tuple], 
                               indices: List[int] = None) -> Dict:
        """
        Compare multiple models using cross-validation.
        Each model is tested on the same folds for fair comparison.
        
        Args:
            model_configs: List of (model_class, config) tuples
            indices: Specific indices to use (if None, use all data)
            
        Returns:
            Dictionary with comparison results
        """
        if self.verbose:
            print(f"\nComparing {len(model_configs)} models with {self.n_splits}-fold CV...")
        
        # Create CV folds (same for all models)
        cv_folds = self.create_cv_folds(indices)
        
        # Store results for all models
        all_results = {}
        comparison_summary = {}
        
        # Test each model
        for model_class, config in model_configs:
            model_name = config.get('name', 'Unknown')
            if self.verbose:
                print(f"\nTesting {model_name}...")
            
            fold_results = []
            all_metrics = defaultdict(list)
            
            # Run each fold
            for fold_idx, (fold_train_indices, fold_val_indices) in enumerate(cv_folds):
                if self.verbose:
                    print(f"  Fold {fold_idx + 1}/{self.n_splits}:", end=" ")
                
                # Get file paths for this fold
                fold_train_images = [self.image_paths[i] for i in fold_train_indices]
                fold_train_masks = [self.mask_paths[i] for i in fold_train_indices]
                fold_val_images = [self.image_paths[i] for i in fold_val_indices]
                fold_val_masks = [self.mask_paths[i] for i in fold_val_indices]
                
                # Train model
                start_time = time.time()
                val_metrics = self.train_single_model(
                    model_class, config, fold_train_images, fold_train_masks,
                    fold_val_images, fold_val_masks
                )
                training_time = time.time() - start_time
                
                # Store results
                val_metrics['training_time'] = training_time
                fold_results.append(val_metrics)
                
                for metric, value in val_metrics.items():
                    if metric != 'training_time':
                        all_metrics[metric].append(value)
                
                if self.verbose:
                    print(f"IoU: {val_metrics['iou']:.4f}")
            
            # Calculate summary for this model
            cv_summary = {}
            for metric, values in all_metrics.items():
                cv_summary[f'{metric}_mean'] = np.mean(values)
                cv_summary[f'{metric}_std'] = np.std(values)
            
            # Store results
            all_results[model_name] = {
                'config': config,
                'fold_results': fold_results,
                'cv_summary': cv_summary
            }
            
            # Add to comparison summary
            comparison_summary[model_name] = cv_summary
            
            if self.verbose:
                mean_iou = cv_summary['iou_mean']
                std_iou = cv_summary['iou_std']
                print(f"  {model_name}: {mean_iou:.4f} ± {std_iou:.4f}")
        
        # Print comparison summary
        if self.verbose:
            print(f"\n{'='*60}")
            print("MODEL COMPARISON SUMMARY")
            print(f"{'='*60}")
            
            # Sort by mean IoU
            sorted_models = sorted(comparison_summary.items(), 
                                 key=lambda x: x[1]['iou_mean'], reverse=True)
            
            for i, (model_name, summary) in enumerate(sorted_models):
                mean_iou = summary['iou_mean']
                std_iou = summary['iou_std']
                print(f"{i+1:2d}. {model_name:25}: {mean_iou:.4f} ± {std_iou:.4f}")
        
        return {
            'individual_results': all_results,
            'comparison_summary': comparison_summary,
            'cv_folds_used': len(cv_folds)
        }


class ModelComparator:
    """Handles model comparison with train/test splits and generalization testing."""
    
    def __init__(self, data_dir: str = "manual_labels", image_type: str = 'W',
                 test_size: float = 0.2, n_splits: int = 5, random_state: int = 42,
                 augmentations_per_image: int = 3, verbose: bool = True):
        """
        Initialize comparator with train/test split for generalization testing.
        """
        self.cv = CrossValidator(data_dir, image_type, n_splits, random_state, 
                               augmentations_per_image, verbose)
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        
        # Create train/test split
        from sklearn.model_selection import train_test_split
        indices = list(range(len(self.cv.image_paths)))
        self.train_indices, self.test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        
        if verbose:
            print(f"Created train/test split: {len(self.train_indices)}/{len(self.test_indices)}")
    
    def run_cv_comparison(self, model_configs: List[Tuple]) -> Dict:
        """Run CV comparison on training set only."""
        return self.cv.compare_multiple_models(model_configs, self.train_indices)
    
    def evaluate_generalization(self, model_configs: List[Tuple]) -> Dict:
        """Train on full training set and test on held-out test set."""
        if self.verbose:
            print(f"\n{'='*60}")
            print("GENERALIZATION EVALUATION")
            print(f"{'='*60}")
        
        device = get_device()
        generalization_results = {}
        
        # Get test data
        test_images = [self.cv.image_paths[i] for i in self.test_indices]
        test_masks = [self.cv.mask_paths[i] for i in self.test_indices]
        
        for model_class, config in model_configs:
            model_name = config.get('name', 'Unknown')
            if self.verbose:
                print(f"\nTraining {model_name} on full training set...")
            
            # Get full training data
            train_images = [self.cv.image_paths[i] for i in self.train_indices]
            train_masks = [self.cv.mask_paths[i] for i in self.train_indices]
            
            # Use train_model() which handles augmentation internally
            start_time = time.time()
            model = model_class(
                n_classes=1, backbone=config['backbone'],
                pretrained=config['pretrained'], use_attention=config['use_attention']
            ).to(device)
            
            criterion = get_loss_function(config)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 1e-5)
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
            )
            
            # Simplified config for generalization training
            gen_config = config.copy()
            gen_config['verbose'] = False
            gen_config['save_plots'] = False
            gen_config['num_epochs'] = config['num_epochs']
            
            # Train using train_model (handles augmentation automatically)
            train_results = train_model(
                model=model,
                train_images=train_images,
                train_masks=train_masks,
                val_images=test_images,
                val_masks=test_masks,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=gen_config['num_epochs'],
                device=device,
                config=gen_config,
                augmentations_per_image=self.cv.augmentations_per_image,
                save_plots=False
            )
            
            training_time = time.time() - start_time
            test_metrics = train_results['final_val_metrics']  # Fixed variable name
            best_train_iou = train_results['best_iou']
            
            # Store results
            generalization_results[model_name] = {
                'final_train_iou': best_train_iou,
                'test_metrics': dict(test_metrics),  # Fixed variable name
                'training_time': training_time
            }
            
            if self.verbose:
                print(f"  Test IoU: {test_metrics['iou']:.4f}")  # Fixed variable name
        
        # Print summary
        if self.verbose:
            print(f"\n{'='*60}")
            print("GENERALIZATION SUMMARY")
            print(f"{'='*60}")
            
            sorted_results = sorted(generalization_results.items(), 
                                  key=lambda x: x[1]['test_metrics']['iou'], reverse=True)
            
            for i, (model_name, results) in enumerate(sorted_results):
                test_iou = results['test_metrics']['iou']
                train_iou = results['final_train_iou']
                overfitting = train_iou - test_iou
                print(f"{i+1:2d}. {model_name:25}: Test IoU = {test_iou:.4f}, "
                      f"Overfitting = {overfitting:.4f}")
        
        return generalization_results


# Convenience functions
def quick_cv(model_class: Callable, config: Dict, data_dir: str = "manual_labels", 
             image_type: str = 'W', n_splits: int = 5, augmentations_per_image: int = 3) -> Dict:
    """Quick single model cross-validation."""
    cv = CrossValidator(data_dir=data_dir, image_type=image_type, n_splits=n_splits, 
                       augmentations_per_image=augmentations_per_image)
    return cv.cross_validate_single_model(model_class, config)


def quick_model_comparison(model_configs: List[Tuple], data_dir: str = "manual_labels",
                          image_type: str = 'W', n_splits: int = 5, 
                          augmentations_per_image: int = 3, include_generalization: bool = False) -> Dict:
    """Quick multi-model comparison."""
    if include_generalization:
        comparator = ModelComparator(data_dir=data_dir, image_type=image_type, n_splits=n_splits,
                                   augmentations_per_image=augmentations_per_image)
        cv_results = comparator.run_cv_comparison(model_configs)
        gen_results = comparator.evaluate_generalization(model_configs)
        return {'cv_results': cv_results, 'generalization_results': gen_results}
    else:
        cv = CrossValidator(data_dir=data_dir, image_type=image_type, n_splits=n_splits,
                           augmentations_per_image=augmentations_per_image)
        return cv.compare_multiple_models(model_configs)


if __name__ == "__main__":
    print("Cross-validation module loaded successfully!")
    print("Use CrossValidator for simple CV on all data")
    print("Use ModelComparator for CV + generalization testing")
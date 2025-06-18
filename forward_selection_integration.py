"""
Forward Selection for Augmentation Discovery - Integrated with Existing Code
"""

import os
import torch
import numpy as np
import albumentations as A
import cv2
from sklearn.model_selection import KFold
from tqdm import tqdm
import json
import copy
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Import your existing modules
from dataset import CellSegmentationDataset
from advanced_models import UNetWithBackbone
from losses import get_loss_function
from utils import get_device, calculate_metrics, EarlyStopping
from train import train_epoch, evaluate


class AugmentationSelector:
    """
    Forward selection for augmentation strategies using existing codebase.
    """
    
    def __init__(self, 
                 base_config: Dict,
                 data_dir: str,
                 improvement_threshold: float = 0.005,
                 max_augmentations: int = 8,
                 cv_folds: int = 3,
                 quick_evaluation: bool = True):
        """
        Args:
            base_config: Your existing config dictionary from main.py
            data_dir: Path to your data directory
            improvement_threshold: Minimum IoU improvement to continue
            max_augmentations: Maximum augmentations to select
            cv_folds: Number of CV folds
            quick_evaluation: Use faster evaluation for selection
        """
        self.base_config = base_config
        self.data_dir = data_dir
        self.improvement_threshold = improvement_threshold
        self.max_augmentations = max_augmentations
        self.cv_folds = cv_folds
        self.quick_evaluation = quick_evaluation
        
        # Create augmentation candidates
        self.augmentation_candidates = self._create_augmentation_candidates()
        self.base_names = []  # Track base names for grouping
        
        # Load dataset paths once
        self._load_dataset_paths()

        self.selection_history = []
        
    def _create_augmentation_candidates(self) -> Dict[str, A.BasicTransform]:
        """
        Create non-redundant augmentation transforms for methods showing ≥0.5% improvement.
        Redundancies removed:
        - Flip (use HorizontalFlip + VerticalFlip separately for more control)
        - RandomBrightness (covered by ColorJitter)
        - GaussianBlur (covered by AdvancedBlur)
        - HueSaturationValue (mostly for color images, we have grayscale)
        """
        candidates = {
            # === GEOMETRIC TRANSFORMATIONS (7) ===
            'horizontal_flip': A.HorizontalFlip(p=1.0),  # -0.0161
            
            'affine': A.Affine(  # -0.0149 (most comprehensive geometric)
                scale=(0.95, 1.05),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
                shear=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT,
                p=1.0
            ),
            
            'random_rotate_90': A.RandomRotate90(p=1.0),  # -0.0136 (discrete rotations)
            
            'transpose': A.Transpose(p=1.0),  # -0.0127 (diagonal flip)
            
            # Skip 'flip' - redundant with horizontal_flip + vertical_flip
            # Skip 'rotate' - covered by affine
            
            'grid_distortion': A.GridDistortion(  # -0.0117 (elastic deformation)
                num_steps=5,
                distort_limit=0.3,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT,
                p=1.0
            ),
            
            'vertical_flip': A.VerticalFlip(p=1.0),  # -0.0102
            
            'optical_distortion': A.OpticalDistortion(  # -0.0056 (lens-like distortion)
                distort_limit=0.5,
                shift_limit=0.5,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT,
                p=1.0
            ),
            
            'gauss_noise': A.GaussNoise(  # -0.0099 (standard Gaussian noise)
                var_limit=(10.0, 50.0),
                mean=0,
                per_channel=True,
                p=1.0
            ),
            
            # === INTENSITY TRANSFORMATIONS (4) ===
            'invert': A.InvertImg(p=1.0),  # -0.0104 (critical for microscopy)
            
            'solarize': A.Solarize(  # -0.0087 (unique threshold effect)
                threshold=128,
                p=1.0
            ),
            
            # Skip 'random_brightness' - covered by color_jitter
            # Skip 'hue_saturation_value' - for grayscale, only V matters (covered by gamma)
            
            'random_gamma': A.RandomGamma(  # -0.0052 (non-linear intensity)
                gamma_limit=(80, 120),
                p=1.0
            ),
            
            'color_jitter': A.ColorJitter(  # -0.0062 (comprehensive color/brightness)
                brightness=0.2,
                contrast=0.2,
                saturation=0,  # Set to 0 for grayscale
                hue=0,         # Set to 0 for grayscale
                p=1.0
            ),
            
            # === BLUR/FILTER (1) ===
            # Skip 'gaussian_blur' - covered by advanced_blur
            'advanced_blur': A.AdvancedBlur(  # -0.0061 (includes Gaussian + motion blur)
                blur_limit=(3, 7),
                sigmaX_limit=(0.2, 1.0),
                sigmaY_limit=(0.2, 1.0),
                rotate_limit=90,
                beta_limit=(0.5, 8.0),
                noise_limit=(0.9, 1.1),
                p=1.0
            ),
            
            # === SCALE TRANSFORMATION (1) ===
            'downscale': A.Downscale(  # -0.0074 (resolution degradation)
                scale_min=0.5,
                scale_max=0.75,
                interpolation=cv2.INTER_LINEAR,
                p=1.0
            ),
            
            # === DROPOUT METHODS (2) ===
            'coarse_dropout': A.CoarseDropout(  # -0.0073 (rectangular dropouts)
                max_holes=8,
                max_height=8,
                max_width=8,
                min_holes=4,
                min_height=4,
                min_width=4,
                fill_value=0,
                p=1.0
            ),
            
            'grid_dropout': A.GridDropout(ratio=0.1, unit_size_min=4, unit_size_max=8, random_offset=True, p=0.5),
            
            # === CROPPING (2) ===
            'random_crop': A.RandomCrop(  # -0.0087 (spatial crop)
                height=96,  # 75% of 128
                width=96,   # 75% of 128
                p=1.0
            ),
            
            'crop_and_pad': A.CropAndPad(  # -0.0059 (crop with padding)
                percent=(-0.1, 0.1),
                pad_mode=cv2.BORDER_REFLECT,
                p=1.0
            ),
        }
        
        return candidates
    
    def _load_dataset_paths(self):
        """
        Load original, unprocessed dataset paths for forward selection.
        We want to test augmentations on the raw data, not pre-augmented data.
        """
        # Load from the original directories (same logic as your prepare_data function)
        images_dir = os.path.join(self.data_dir, "Labelled_images")
        masks_dir = os.path.join(self.data_dir, "GT_masks")
        
        # Verify directories exist
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            raise FileNotFoundError(f"Original data directories not found in {self.data_dir}")
        
        # Get all files
        all_masks = sorted(os.listdir(masks_dir))
        all_images = sorted(os.listdir(images_dir))
        
        self.image_paths = []
        self.mask_paths = []
        self.base_names = []  # Track base names for grouping
        
        # Find matching image-mask pairs (same logic as your dataset.py)
        for mask_file in all_masks:
            # Check if mask file ends with GT.tif
            if not mask_file.endswith('GT.tif'):
                continue
            
            # Extract image type (B or W) - same logic as your prepare_data
            parts = mask_file.split('_')
            img_type = parts[3][1]  # 1B or 1W
            
            # Filter by image type if specified
            if img_type != self.base_config['image_type']:
                continue
            
            # Find corresponding original image
            original_file = mask_file[:-7] + '.tif'  # Replace _GT.tif with .tif
            
            if original_file in all_images:
                img_path = os.path.join(images_dir, original_file)
                mask_path = os.path.join(masks_dir, mask_file)
                
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
                
                # Extract base name for grouping (remove _GT.tif and file extension)
                base_name = mask_file[:-7]  # Remove _GT.tif
                self.base_names.append(base_name)
        
        print(f"Loaded {len(self.image_paths)} original {self.base_config['image_type']} images for selection")
        print(f"Found {len(set(self.base_names))} unique base images")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No matching image-mask pairs found for image type '{self.base_config['image_type']}'!")
    
    def _create_augmentation_pipeline(self, selected_augs: List[str], for_selection: bool = True) -> A.Compose:
        """
        Create Albumentations pipeline from selected augmentations.
        
        Args:
            selected_augs: List of augmentation names to include
            for_selection: If True, use p=0.7 for testing with diversity.
                          If False, use realistic probabilities for production.
        """
        if not selected_augs:
            return A.Compose([])
        
        transforms = []
        for aug_name in selected_augs:
            aug = copy.deepcopy(self.augmentation_candidates[aug_name])
            
            if for_selection:
                # For forward selection: Use p=0.7 for natural diversity
                # Each augmentation has 70% chance to be applied
                aug.p = 0.7
            else:
                # For production: Use realistic probabilities
                if aug_name in ['horizontal_flip', 'vertical_flip']:
                    aug.p = 0.5
                elif aug_name == 'clahe':
                    aug.p = 0.7  # Higher probability for critical microscopy enhancement
                elif aug_name in ['random_rotate_90', 'transpose']:
                    aug.p = 0.3  # Moderate for geometric
                else:
                    aug.p = 0.3  # Conservative for others
                
            transforms.append(aug)
        
        return A.Compose(transforms)
    
    def _create_augmented_dataset(self, image_paths: List[str], mask_paths: List[str], 
                            aug_pipeline: A.Compose, augmentations_per_image: int = 2):
        """
        Apply augmentation pipeline to create expanded dataset with smart diversity.
        
        Instead of applying the same full pipeline to create identical copies,
        we use a random subset strategy for better diversity.
        """
        if len(aug_pipeline.transforms) == 0:
            return image_paths, mask_paths
        
        expanded_images = image_paths.copy()
        expanded_masks = mask_paths.copy()
        
        # Extract individual augmentations from the pipeline
        selected_augs = aug_pipeline.transforms
        
        for img_path, mask_path in zip(image_paths, mask_paths):
            # Load original image and mask
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8) * 255
            
            # Create diverse augmented versions
            for aug_idx in range(augmentations_per_image):
                # Option 1: Apply full pipeline (relies on p=0.7 for each transform)
                # This is actually fine since each transform has p=0.7, creating natural diversity
                augmented = aug_pipeline(image=image, mask=mask)
                
                # Option 2: Random subset strategy (if you want more control)
                # if len(selected_augs) > 1:
                #     # Randomly select subset of augmentations
                #     n_augs = np.random.randint(1, len(selected_augs) + 1)
                #     subset_augs = np.random.choice(selected_augs, n_augs, replace=False)
                #     subset_pipeline = A.Compose(list(subset_augs))
                #     augmented = subset_pipeline(image=image, mask=mask)
                # else:
                #     augmented = aug_pipeline(image=image, mask=mask)
                
                # Save augmented version
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                temp_img_path = f"{base_name}_aug_{aug_idx}.tif"
                temp_mask_path = f"{base_name}_mask_{aug_idx}.tif"
                
                os.makedirs("temp_selection", exist_ok=True)
                cv2.imwrite(os.path.join("temp_selection", temp_img_path), augmented['image'])
                cv2.imwrite(os.path.join("temp_selection", temp_mask_path), augmented['mask'])
                
                expanded_images.append(os.path.join("temp_selection", temp_img_path))
                expanded_masks.append(os.path.join("temp_selection", temp_mask_path))
        
        return expanded_images, expanded_masks
    
    def _train_and_evaluate_fold(self, train_imgs: List[str], train_masks: List[str],
                           val_imgs: List[str], val_masks: List[str],
                           aug_pipeline: A.Compose) -> float:
        """Train and evaluate model for one fold."""
        
        # Create augmented training data
        if len(aug_pipeline.transforms) > 0:
            train_imgs_aug, train_masks_aug = self._create_augmented_dataset(
                train_imgs, train_masks, aug_pipeline, augmentations_per_image=5
            )
            original_count = len(train_imgs)
            augmented_count = len(train_imgs_aug) - original_count
            # Just show the final dataset size
            print(f"      Training with {len(train_imgs_aug)} images ({original_count} original + {augmented_count} augmented)")
        else:
            train_imgs_aug, train_masks_aug = train_imgs, train_masks
            print(f"      Training with {len(train_imgs_aug)} original images (no augmentation)")
        
        # Create datasets using your existing class
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
        train_loader = DataLoader(train_dataset, batch_size=self.base_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.base_config['batch_size'], shuffle=False)
        
        # Create model using your existing class
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
            optimizer, mode='max', factor=0.5, patience=3, threshold=0.01, min_lr=1e-6
        )
        
        # Early stopping for quick evaluation
        early_stopping = EarlyStopping(
            patience=3 if self.quick_evaluation else 10,
            min_delta=0.01
        )
        
        # Training loop (reduced epochs for quick evaluation)
        max_epochs = 15 if self.quick_evaluation else self.base_config['num_epochs']
        best_iou = 0.0
        
        # Simplified progress tracking
        for epoch in range(max_epochs):
            # Train one epoch using your existing function
            train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
            
            # Evaluate using your existing function  
            val_metrics = evaluate(model, val_loader, device, criterion)
            
            # Update scheduler
            scheduler.step(val_metrics['iou'])
            
            # Track best IoU
            if val_metrics['iou'] > best_iou:
                best_iou = val_metrics['iou']
            
            # Early stopping
            if early_stopping.step(val_metrics['iou']):
                print(f"      Stopped at epoch {epoch+1}/{max_epochs} - Best IoU: {best_iou:.4f}")
                break
            
            # Only print final result unless it's taking long
            if epoch == max_epochs - 1:
                print(f"      Completed {epoch+1} epochs - Best IoU: {best_iou:.4f}")
        
        # Cleanup temporary files
        self._cleanup_temp_files()
        
        return best_iou
    
    def _cleanup_temp_files(self):
        """Clean up temporary augmentation files."""
        import shutil
        if os.path.exists("temp_selection"):
            shutil.rmtree("temp_selection")


    def _evaluate_augmentation_set(self, selected_augs: List[str]) -> float:
        """Evaluate a set of augmentations using cross-validation with proper grouping."""
        
        # Simplified header
        if selected_augs:
            print(f"\n  Testing: {' + '.join(selected_augs)}")
        else:
            print(f"\n  Testing: Baseline (no augmentation)")
        
        # Use for_selection=True to set p=0.7 for natural diversity
        aug_pipeline = self._create_augmentation_pipeline(selected_augs, for_selection=True)
        
        # Use GroupKFold to ensure base images stay together
        from sklearn.model_selection import GroupKFold
        group_kfold = GroupKFold(n_splits=self.cv_folds)
        scores = []
        
        # Create group mapping for consistent splitting
        unique_bases = sorted(list(set(self.base_names)))
        base_to_idx = {base: idx for idx, base in enumerate(unique_bases)}
        group_indices = np.array([base_to_idx[base] for base in self.base_names])
        
        # Create dummy X for GroupKFold (we only care about groups)
        X = np.arange(len(self.image_paths))
        
        # Split by groups, ensuring base images stay together
        for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, groups=group_indices)):
            train_bases = set([self.base_names[i] for i in train_idx])
            val_bases = set([self.base_names[i] for i in val_idx])
            
            print(f"    Fold {fold + 1}/{self.cv_folds}:")
            
            # Split data by base image groups
            train_imgs = [self.image_paths[i] for i in train_idx]
            train_masks = [self.mask_paths[i] for i in train_idx]
            val_imgs = [self.image_paths[i] for i in val_idx]
            val_masks = [self.mask_paths[i] for i in val_idx]
            
            # Verify no data leakage
            assert len(train_bases.intersection(val_bases)) == 0, "Data leakage detected!"
            
            # Train and evaluate
            fold_score = self._train_and_evaluate_fold(
                train_imgs, train_masks, val_imgs, val_masks, aug_pipeline
            )
            scores.append(fold_score)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"  Result: {mean_score:.4f} ± {std_score:.4f}")
        
        return mean_score


    def run_forward_selection(self, verbose: bool = True) -> Dict:
        """
        Run the forward selection algorithm.
        
        Note: During selection, we use p=0.7 for all augmentations to create
        natural diversity while still testing their effect. Original images
        are always included in the training set.
        """
        selected_augmentations = []
        remaining_candidates = list(self.augmentation_candidates.keys())
        
        if verbose:
            print("\n" + "="*60)
            print("AUGMENTATION FORWARD SELECTION")
            print("="*60)
            print(f"Dataset: {len(self.image_paths)} images ({self.base_config['image_type']} type)")
            print(f"Model: {self.base_config['backbone']}")
            print(f"Validation: {self.cv_folds}-fold cross-validation")
            print(f"Candidates: {', '.join(remaining_candidates)}")
            print("="*60)
        
        # Baseline performance (no augmentation)
        print("\nEvaluating baseline performance...")
        baseline_score = self._evaluate_augmentation_set([])
        current_best_score = baseline_score
        
        if verbose:
            print(f"\nBaseline IoU: {baseline_score:.4f}")
            print("-"*60)
        
        # Forward selection iterations
        for iteration in range(self.max_augmentations):
            if verbose:
                print(f"\nITERATION {iteration + 1}/{self.max_augmentations}")
                if selected_augmentations:
                    print(f"Current selection: {' + '.join(selected_augmentations)}")
                print(f"Current best IoU: {current_best_score:.4f}")
                print(f"Testing {len(remaining_candidates)} candidates...")
            
            best_candidate = None
            best_score = current_best_score
            candidate_scores = {}
            
            # Test each remaining candidate
            for i, candidate in enumerate(remaining_candidates):
                if verbose:
                    print(f"\n  [{i+1}/{len(remaining_candidates)}] {candidate}:", end="", flush=True)
                
                test_set = selected_augmentations + [candidate]
                score = self._evaluate_augmentation_set(test_set)
                candidate_scores[candidate] = score
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    if verbose:
                        improvement = score - current_best_score
                        print(f"    ✓ New best! (+{improvement:.4f})")
                else:
                    if verbose:
                        change = score - current_best_score
                        print(f"    {change:+.4f}")
            
            # Check improvement
            improvement = best_score - current_best_score
            
            if verbose:
                print(f"\nIteration {iteration + 1} summary:")
                
                # Show top 3 candidates
                sorted_candidates = sorted(candidate_scores.items(), 
                                        key=lambda x: x[1], reverse=True)
                print("  Top candidates:")
                for i, (name, score) in enumerate(sorted_candidates[:3]):
                    change = score - current_best_score
                    marker = "→" if i == 0 else " "
                    print(f"  {marker} {name}: {score:.4f} ({change:+.4f})")
            
            # Stopping criteria
            if improvement < self.improvement_threshold:
                if verbose:
                    print(f"\nStopping: Improvement ({improvement:.4f}) below threshold ({self.improvement_threshold:.4f})")
                break
            
            if best_candidate is None:
                if verbose:
                    print("\nStopping: No improvement found")
                break
            
            # Add best candidate
            selected_augmentations.append(best_candidate)
            remaining_candidates.remove(best_candidate)
            current_best_score = best_score
            
            # Record history
            self.selection_history.append({
                'iteration': iteration + 1,
                'selected': best_candidate,
                'current_set': selected_augmentations.copy(),
                'score': best_score,
                'improvement': improvement,
                'candidate_scores': candidate_scores.copy()
            })
            
            if verbose:
                print(f"\n✓ Added: {best_candidate}")
                print("-"*60)
        
        # Final results
        results = {
            'selected_augmentations': selected_augmentations,
            'final_score': current_best_score,
            'baseline_score': baseline_score,
            'total_improvement': current_best_score - baseline_score,
            'selection_history': self.selection_history,
            'config': self.base_config
        }
        
        if verbose:
            print("\n" + "="*60)
            print("SELECTION COMPLETE")
            print("="*60)
            print(f"Selected augmentations: {' + '.join(selected_augmentations) if selected_augmentations else 'None'}")
            print(f"Final IoU: {current_best_score:.4f}")
            print(f"Improvement: {current_best_score - baseline_score:.4f} ({((current_best_score - baseline_score) / baseline_score * 100):+.1f}%)")
            print("="*60)
        
        return results
    
    def create_optimal_pipeline(self, selected_augs: List[str]) -> A.Compose:
        """
        Create the optimal augmentation pipeline for production use.
        Uses realistic probabilities (not p=1.0 like during selection).
        """
        return self._create_augmentation_pipeline(selected_augs, for_selection=False)


def run_augmentation_selection_experiment():
    """
    Main function to run the augmentation selection experiment using existing config.
    """
    
    # Use your existing config system
    def get_selection_config():
        """Optimized config for augmentation selection."""
        config = {
            'name': 'augmentation_selection',
            'model_type': 'unet',
            'image_type': 'W',  # Or 'B' for fluorescent
            'backbone': 'resnet34',  # Faster than resnet50 for selection
            'use_attention': False,  # Disable for speed during selection
            'batch_size': 2,
            'img_size': (128, 128),
            'num_epochs': 25,  # Reduced for faster selection
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'pretrained': True,
            'seed': 42,
            'loss_fn': 'focal',
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
        }
        return config
    
    # Get configuration
    config = get_selection_config()
    data_dir = 'manual_labels'  # Update to your data directory
    
    print(f"Starting augmentation selection experiment")
    print(f"Image type: {config['image_type']}")
    print(f"Model: {config['backbone']} with attention: {config['use_attention']}")
    
    # Initialize selector
    selector = AugmentationSelector(
        base_config=config,
        data_dir=data_dir,
        improvement_threshold=0.005,  # 0.5% improvement threshold
        max_augmentations=10,
        cv_folds=3,
        quick_evaluation=False  # Faster evaluation for selection
    )
    
    # Run selection
    results = selector.run_forward_selection(verbose=True)
    
    # Save results
    os.makedirs('experiments/augmentation_selection_results', exist_ok=True)
    with open('experiments/augmentation_selection_results/results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create optimal pipeline for production
    optimal_pipeline = selector.create_optimal_pipeline(results['selected_augmentations'])
    
    print(f"\nOptimal augmentation pipeline created!")
    print(f"Use this in your production training:")
    print(f"selected_augs = {results['selected_augmentations']}")
    
    return results, optimal_pipeline


if __name__ == "__main__":
    # Run the experiment
    results, optimal_pipeline = run_augmentation_selection_experiment()
"""
Simplified Dataset Module for Cell Segmentation
Just the dataset class and simple data loading functions
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import albumentations as A
import tempfile
import shutil


class CellSegmentationDataset(Dataset):
    """Dataset class for cell segmentation with microscopy normalization."""
    
    def __init__(self, image_paths: List[str], mask_paths: List[str], 
                 img_size: Tuple[int, int] = (256, 256), normalize: bool = True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.normalize = normalize
        
    def __len__(self):
        return len(self.image_paths)
    
    def normalize_microscopy_image(self, image):
        """Apply advanced normalization for microscopy images."""
        # Remove outliers by clipping extreme values
        p_low, p_high = np.percentile(image, [2, 98])
        image_clipped = np.clip(image, p_low, p_high)
        
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_clahe = clahe.apply(image_clipped.astype(np.uint8))
        
        # Normalize to [0, 1] range
        image_norm = (image_clahe - image_clahe.min()) / (image_clahe.max() - image_clahe.min() + 1e-8)
        return image_norm
    
    def __getitem__(self, idx):
        # Load image and mask
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Resize to the specified size
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # Apply normalization
        if self.normalize:
            image = self.normalize_microscopy_image(image)
        else:
            image = image.astype(np.float32) / 255.0
        
        # Ensure mask is binary (0 or 1)
        mask = (mask > 0).astype(np.float32)
        
        # Convert to tensor format (add channel dimension)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask


def load_original_data(data_dir: str = "manual_labels", image_type: str = 'W') -> Dict:
    """
    Load original (non-augmented) data paths.
    
    Args:
        data_dir: Directory containing Labelled_images and GT_masks
        image_type: 'B' for fluorescent, 'W' for broadband
        
    Returns:
        Dictionary with image_paths and mask_paths lists
    """
    images_dir = os.path.join(data_dir, "Labelled_images")
    masks_dir = os.path.join(data_dir, "GT_masks")
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Data directories not found in {data_dir}")
    
    all_masks = sorted(os.listdir(masks_dir))
    all_images = sorted(os.listdir(images_dir))
    
    image_paths = []
    mask_paths = []
    
    for mask_file in all_masks:
        if not mask_file.endswith('GT.tif'):
            continue
        
        # Extract image type (B or W)
        parts = mask_file.split('_')
        img_type = parts[3][1]  # 1B or 1W
        
        if img_type != image_type:
            continue
        
        # Find corresponding original image
        original_file = mask_file[:-7] + '.tif'  # Replace _GT.tif with .tif
        
        if original_file in all_images:
            img_path = os.path.join(images_dir, original_file)
            mask_path = os.path.join(masks_dir, mask_file)
            
            image_paths.append(img_path)
            mask_paths.append(mask_path)
    
    print(f"Loaded {len(image_paths)} original {image_type} images")
    
    return {
        'image_paths': image_paths,
        'mask_paths': mask_paths
    }


def prepare_data(image_paths: List[str], mask_paths: List[str], 
                batch_size: int = 2, img_size: Tuple[int, int] = (256, 256),
                shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader from image and mask paths.
    
    Args:
        image_paths: List of image file paths
        mask_paths: List of mask file paths
        batch_size: Batch size for DataLoader
        img_size: Target image size (height, width)
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader ready for training/evaluation
    """
    dataset = CellSegmentationDataset(image_paths, mask_paths, img_size=img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class CellAugmenter:
    """Simple augmenter using the 4 best augmentations from forward selection."""
    
    def __init__(self, augmentations_per_image: int = 3):
        self.augmentations_per_image = augmentations_per_image
        self.temp_dir = "temp_augmentation"  # Fixed directory name
        
        # Create the optimal pipeline
        self.pipeline = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05),
                     rotate=(-15, 15), shear=(-5, 5), p=0.3),
            A.VerticalFlip(p=0.5),
            A.AdvancedBlur(blur_limit=(3, 7), p=0.3)
        ])
    
    def augment_training_data(self, train_images: List[str], train_masks: List[str]) -> Tuple[List[str], List[str]]:
        """
        Augment training data and return expanded lists.
        Returns: (all_train_images, all_train_masks) including originals + augmented
        """
        if self.augmentations_per_image == 0:
            return train_images, train_masks
        
        # Create/clean temp directory
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Start with originals
        all_images = train_images.copy()
        all_masks = train_masks.copy()
        
        # Add augmented versions
        for img_path, mask_path in zip(train_images, train_masks):
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                print(f"Warning: Could not load {img_path} or {mask_path}")
                continue
                
            mask = (mask > 0).astype(np.uint8) * 255
            
            for i in range(self.augmentations_per_image):
                # Apply augmentation
                aug_result = self.pipeline(image=image, mask=mask)
                
                # Save augmented files
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                aug_img_path = os.path.join(self.temp_dir, f"{base_name}_aug{i}.tif")
                aug_mask_path = os.path.join(self.temp_dir, f"{base_name}_mask{i}.tif")
                
                # Ensure the images are saved successfully
                success_img = cv2.imwrite(aug_img_path, aug_result['image'])
                success_mask = cv2.imwrite(aug_mask_path, aug_result['mask'])
                
                if success_img and success_mask:
                    all_images.append(aug_img_path)
                    all_masks.append(aug_mask_path)
                else:
                    print(f"Warning: Failed to save augmented files for {base_name}")
        
        print(f"Training data: {len(train_images)} original + {len(all_images) - len(train_images)} augmented = {len(all_images)} total")
        return all_images, all_masks
    
    def cleanup(self):
        """Remove temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def __enter__(self):
        return self


if __name__ == "__main__":
    # Example usage
    data = load_original_data(image_type='W')
    loader = prepare_data(data['image_paths'], data['mask_paths'], batch_size=2)
    print(f"Created DataLoader with {len(loader)} batches")
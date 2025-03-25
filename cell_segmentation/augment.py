import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import albumentations as A

import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import albumentations as A

def generate_albumentations_cell_dataset(image_paths, mask_paths, output_dir, augmentations_per_image=5):
    """
    Generate augmented cell microscopy images using Albumentations library,
    with grayscale-compatible transforms.
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    
    # Define the augmentation pipeline for microscopy images - GRAYSCALE COMPATIBLE
    microscopy_aug = A.Compose([
        # SPATIAL TRANSFORMS - Even more conservative
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(
            scale_limit=0.05,
            rotate_limit=5,
            shift_limit=0.05,
            p=0.3,
            border_mode=cv2.BORDER_CONSTANT,
            value=80
        ),
        
        # REPLACE problematic transforms with grayscale-compatible alternatives
        A.OneOf([
            # Grayscale-compatible blurring options
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
            # Small localized changes for grayscale
            A.GridDropout(ratio=0.1, unit_size_min=4, unit_size_max=8, random_offset=True, p=0.5)
        ], p=0.3),
        
        # CELL ENHANCEMENT - Critical for microscopy
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.7),
        
        # Better intensity adjustments
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.2,
            p=0.7
        ),
        
        # NOISE HANDLING - Grayscale compatible
        A.GaussNoise(var_limit=(3, 10), p=0.3),  # Only use GaussNoise, remove ISONoise
        
        # CELL DETAIL ENHANCEMENT
        A.Sharpen(alpha=(0.2, 0.3), lightness=(0.7, 1.0), p=0.4),  # Combining in a single option
        
        # Background uniformity
        A.RandomGamma(gamma_limit=(90, 110), p=0.3)
    ])
    
    # Process each image-mask pair
    new_image_paths = []
    new_mask_paths = []
    
    for idx, (img_path, mask_path) in enumerate(tqdm(zip(image_paths, mask_paths), total=len(image_paths))):
        # Load original image and mask
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure mask is binary (0 or 255)
        mask = (mask > 0).astype(np.uint8) * 255
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Save the original image and mask
        orig_img_path = os.path.join(output_dir, 'images', f"{base_name}_orig.tif")
        orig_mask_path = os.path.join(output_dir, 'masks', f"{base_name}_orig.tif")
        cv2.imwrite(orig_img_path, image)
        cv2.imwrite(orig_mask_path, mask)
        new_image_paths.append(orig_img_path)
        new_mask_paths.append(orig_mask_path)
        
        # Generate augmented versions
        for aug_idx in range(augmentations_per_image):
            # Apply augmentation
            augmented = microscopy_aug(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            
            # Save augmented image and mask
            aug_img_path = os.path.join(output_dir, 'images', f"{base_name}_aug{aug_idx}.tif")
            aug_mask_path = os.path.join(output_dir, 'masks', f"{base_name}_aug{aug_idx}.tif")
            cv2.imwrite(aug_img_path, aug_image)
            cv2.imwrite(aug_mask_path, aug_mask)
            new_image_paths.append(aug_img_path)
            new_mask_paths.append(aug_mask_path)
    
    print(f"Generated dataset with {len(new_image_paths)} images " 
          f"({len(image_paths)} original + {len(new_image_paths) - len(image_paths)} augmented)")
    
    return new_image_paths, new_mask_paths

def create_augmented_dataset(data_dir, image_type='both', aug_per_image=5):
    """
    Create an augmented dataset from original images using Albumentations.
    
    Args:
        data_dir: Path to manual_labels directory
        image_type: 'B' for fluorescent, 'W' for broadband, or 'both'
        aug_per_image: Number of augmented versions to create per original image
    """
    if image_type == 'both':
        # Process both image types
        for img_type in ['B', 'W']:
            print(f"Processing {img_type} images...")
            create_augmented_dataset(data_dir, img_type, aug_per_image)
        return
    
    # Paths to directories
    masks_dir = os.path.join(data_dir, "GT_masks")
    images_dir = os.path.join(data_dir, "Labelled_images")
    
    # Find matching image-mask pairs
    image_paths, mask_paths = [], []
    all_masks = sorted(os.listdir(masks_dir))
    all_images = sorted(os.listdir(images_dir))
    
    for mask_file in all_masks:
        if not mask_file.endswith('GT.tif'):
            continue
        
        # Extract image type
        parts = mask_file.split('_')
        img_type = parts[3][1]  # 1B or 1W
        
        if img_type != image_type:
            continue
        
        # Find corresponding original image
        original_file = mask_file[:-7] + '.tif'
        
        if original_file in all_images:
            image_paths.append(os.path.join(images_dir, original_file))
            mask_paths.append(os.path.join(masks_dir, mask_file))
    
    print(f"Found {len(image_paths)} original {image_type} images with matching ground truth masks")
    
    # Create output directory for augmented dataset
    output_dir = os.path.join(data_dir, f"augmented_{image_type}")
    
    # Generate augmented dataset using Albumentations
    generate_albumentations_cell_dataset(image_paths, mask_paths, output_dir, aug_per_image)
    
    print(f"Augmented dataset created at {output_dir}")
    return output_dir

if __name__ == "__main__":
    # Define data directory
    data_dir = 'data/manual_labels'  # Update this path to your data directory
    
    # Create augmented dataset for both image types
    create_augmented_dataset(data_dir, image_type='both', aug_per_image=5)
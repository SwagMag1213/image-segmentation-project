import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class CellSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=(256, 256), normalize=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.normalize = normalize
        
    def __len__(self):
        return len(self.image_paths)
    
    def normalize_microscopy_image(self, image):
        """Apply advanced normalization for microscopy images"""
        # First, remove outliers by clipping extreme values
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
        
        # Apply advanced normalization if enabled
        if self.normalize:
            image = self.normalize_microscopy_image(image)
        else:
            # Basic normalization to [0, 1] range
            image = image.astype(np.float32) / 255.0
        
        # Ensure mask is binary (0 or 1)
        mask = (mask > 0).astype(np.float32)
        
        # Convert to tensor format (add channel dimension)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask

def prepare_data(data_dir, image_type='both', test_size=0.2, batch_size=2, seed=42, img_size=(256, 256)):
    """
    Prepare data for training and testing.
    
    Args:
        data_dir: Path to manual_labels directory
        image_type: 'B' for fluorescent, 'W' for broadband, or 'both'
        test_size: Fraction of data to use for testing
        batch_size: Batch size for dataloaders
        seed: Random seed for reproducibility
    """
    # Paths to directories
    masks_dir = os.path.join(data_dir, "GT_masks")
    images_dir = os.path.join(data_dir, "Labelled_images")
    
    # Get all files
    all_masks = sorted(os.listdir(masks_dir))
    all_images = sorted(os.listdir(images_dir))
    
    # Store matching image/mask pairs
    image_paths = []
    mask_paths = []
    
    # Find matching pairs
    for mask_file in all_masks:
        # Check if mask file ends with GT.tif
        if not mask_file.endswith('GT.tif'):
            continue
        
        # Extract image type (B or W)
        parts = mask_file.split('_')
        img_type = parts[3][1]  # 1B or 1W
        
        # Filter by image type if specified
        if image_type != 'both' and img_type != image_type:
            continue
        
        # Find corresponding original image
        original_file = mask_file[:-7] + '.tif'  # Replace _GT.tif with .tif
        
        if original_file in all_images:
            image_paths.append(os.path.join(images_dir, original_file))
            mask_paths.append(os.path.join(masks_dir, mask_file))
    
    # Print dataset statistics
    print(f"Found {len(image_paths)} images with matching ground truth masks")
    if image_type == 'both':
        b_count = sum(1 for path in image_paths if '_1B.' in path)
        w_count = sum(1 for path in image_paths if '_1W.' in path)
        print(f"  - {b_count} fluorescent (B) images for NK cells")
        print(f"  - {w_count} broadband (W) images for cancer cells")
    
    # Split into train and test sets
    train_img_paths, test_img_paths, train_mask_paths, test_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=test_size, random_state=seed
    )
    
    print(f"Training set: {len(train_img_paths)} images")
    print(f"Test set: {len(test_img_paths)} images")
    
    # Create datasets
    train_dataset = CellSegmentationDataset(train_img_paths, train_mask_paths, img_size=img_size)
    test_dataset = CellSegmentationDataset(test_img_paths, test_mask_paths, img_size=img_size)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def prepare_augmented_data(data_dir, image_type, test_size=0.2, batch_size=2, seed=42, img_size=(256, 256)):
    """
    Prepares augmented data while preventing data leakage between train and test sets.
    
    Args:
        data_dir: Path to manual_labels directory
        image_type: 'B' for fluorescent, 'W' for broadband
        test_size: Fraction of data to use for testing
        batch_size: Batch size for dataloaders
        seed: Random seed for reproducibility
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
    
    # Group images by their base original image
    image_groups = {}
    for img in all_images:
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
        
        # Add to appropriate group
        if base_name not in image_groups:
            image_groups[base_name] = []
        
        image_groups[base_name].append((
            os.path.join(aug_images_dir, img),
            mask_path
        ))
    
    # Get the base names of all groups
    base_names = list(image_groups.keys())
    
    # Split base names into train and test
    # This is the key step - we split by base name so all related augmentations stay together
    train_bases, test_bases = train_test_split(
        base_names, test_size=test_size, random_state=seed
    )
    
    # Create train and test sets
    train_img_paths = []
    train_mask_paths = []
    test_img_paths = []
    test_mask_paths = []
    
    # Add all images from training base names to train set
    for base in train_bases:
        for img_path, mask_path in image_groups[base]:
            train_img_paths.append(img_path)
            train_mask_paths.append(mask_path)
    
    # Add all images from test base names to test set
    for base in test_bases:
        for img_path, mask_path in image_groups[base]:
            test_img_paths.append(img_path)
            test_mask_paths.append(mask_path)
    
    # Print dataset statistics
    print(f"Found {len(base_names)} original base images with augmentations")
    print(f"Training set: {len(train_img_paths)} images from {len(train_bases)} base images")
    print(f"Test set: {len(test_img_paths)} images from {len(test_bases)} base images")
    
    # Create datasets
    train_dataset = CellSegmentationDataset(train_img_paths, train_mask_paths, img_size=img_size)
    test_dataset = CellSegmentationDataset(test_img_paths, test_mask_paths, img_size=img_size)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
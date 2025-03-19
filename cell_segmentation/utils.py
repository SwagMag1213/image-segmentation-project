import os
import torch
import numpy as np
import torch.nn as nn
from scipy import ndimage

def dice_loss(pred, target, smooth=1.0):
    """
    Calculate Dice loss for segmentation
    
    Args:
        pred: Model predictions
        target: Ground truth masks
        smooth: Smoothing factor to avoid division by zero
    """
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def combo_loss(pred, target, alpha=0.5):
    """
    Combine BCE and Dice loss
    
    Args:
        pred: Model predictions
        target: Ground truth masks
        alpha: Weight for BCE loss
    """
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return alpha * bce + (1-alpha) * dice

def post_process(prediction, min_size=15):
    """
    Remove small objects from binary segmentation mask
    
    Args:
        prediction: Probability map or binary mask
        min_size: Minimum size of object to keep (in pixels)
    """
    # Convert probability map to binary mask (0s and 1s)
    binary = (prediction > 0.5).astype(np.uint8)
    
    # Find all connected components (groups of white pixels)
    labeled, num = ndimage.label(binary)
    
    # Calculate the size (pixel count) of each component
    sizes = ndimage.sum(binary, labeled, range(1, num+1))
    
    # Identify which components are smaller than min_size
    small_objects = sizes < min_size
    
    # Create a mask of pixels to remove
    remove_pixels = small_objects[labeled-1]
    
    # Remove the small objects
    binary[remove_pixels] = 0
    
    return binary

def iou_score(pred, target, apply_post_processing=False, min_size=25):
    """
    Calculate IoU (Intersection over Union) metric
    
    Args:
        pred: Model predictions
        target: Ground truth masks
        apply_post_processing: Whether to apply post-processing
        min_size: Minimum size of object to keep if post-processing
    """
    # Apply sigmoid to convert to probability
    pred_sigmoid = torch.sigmoid(pred)
    
    # Move to CPU and convert to numpy for post-processing
    pred_np = pred_sigmoid.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    if apply_post_processing:
        # Apply post-processing to clean up the prediction
        pred_binary = post_process(pred_np, min_size=min_size)
    else:
        # Standard thresholding at 0.5
        pred_binary = (pred_np > 0.5).astype(np.uint8)
    
    # Convert target to binary format
    target_binary = (target_np > 0.5).astype(np.uint8)
    
    # Calculate IoU
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum((pred_binary + target_binary) > 0)
    
    iou = intersection / (union + 1e-7)
    return iou

def calculate_metrics(pred, target):
    """Calculate additional metrics beyond IoU"""
    # Apply threshold to get binary predictions
    pred_binary = (pred > 0.5).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # Calculate metrics
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
    
    # Avoid division by zero
    epsilon = 1e-7
    
    # Calculate metrics
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'accuracy': accuracy
    }

def get_device():
    """
    Determine the best available device (MPS, CUDA, or CPU)
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device")
    else:
        device = torch.device("cpu")
        print(f"Using CPU")
    
    return device

def ensure_dir(directory):
    """Ensure a directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
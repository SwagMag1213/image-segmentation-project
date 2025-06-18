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

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, current_score):
        if self.best_score is None or current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    

## new code
import os
from typing import Dict
import torch
import yaml
import numpy as np


class PluginManager:
    """
    Plugin manager for MAESTER, which is used to register and get plugins.

    Code is adopted from https://gist.github.com/mepcotterell/6004997
    """

    def __init__(self):
        self.plugin_container: Dict[str : Dict[str:object]] = {}  # type: ignore

    def register_plugin(
        self, plugin_type: str, plugin_name: str, plugin_object: object
    ):
        if plugin_type not in self.plugin_container:
            self.plugin_container[plugin_type] = {}

        self.plugin_container[plugin_type][plugin_name] = plugin_object

    def get(self, plugin_type: str, plugin_name: str):
        return self.plugin_container[plugin_type][plugin_name]


def register_plugin(plugin_type: str, plugin_name: str):
    def decorator(cls):
        plugin_manager.register_plugin(plugin_type, plugin_name, cls)
        return cls

    return decorator


def get_plugin(plugin_type: str, plugin_name: str):
    return plugin_manager.get(plugin_type, plugin_name)


def read_yaml(path) -> Dict:
    """
    Helper function to read yaml file
    """
    file = open(path, "r", encoding="utf-8")
    string = file.read()
    data_dict = yaml.safe_load(string)

    return data_dict


def save_checkpoint(path, state_dict, name):
    filename = os.path.join(path, name)
    torch.save(state_dict, filename)
    print("Saving checkpoint:", filename)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    code is adopted from https://github.com/facebookresearch/mae
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


plugin_manager = PluginManager()
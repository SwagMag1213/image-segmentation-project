"""
Comprehensive Loss Functions for Semantic Segmentation
Based on "A survey of loss functions for semantic segmentation" by Shruti Jadon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


class DiceLoss(nn.Module):
    """Dice Loss - Good for class imbalance"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class BCELoss(nn.Module):
    """Binary Cross-Entropy Loss"""
    def __init__(self):
        super(BCELoss, self).__init__()
        
    def forward(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy - For skewed datasets"""
    def __init__(self, beta=1.0):
        super(WeightedBCELoss, self).__init__()
        self.beta = beta
        
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weights = target * self.beta + (1 - target)
        weighted_bce = (bce * weights).mean()
        return weighted_bce


class BalancedBCELoss(nn.Module):
    """Balanced Cross-Entropy - Weights both positive and negative examples"""
    def __init__(self):
        super(BalancedBCELoss, self).__init__()
        
    def forward(self, pred, target):
        # Calculate beta as 1 - (positive pixels / total pixels)
        beta = 1 - target.sum() / target.numel()
        
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weights = target * beta + (1 - target) * (1 - beta)
        balanced_bce = (bce * weights).mean()
        return balanced_bce


class FocalLoss(nn.Module):
    """Focal Loss - For highly imbalanced datasets"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """Tversky Loss - Generalization of Dice with FP/FN control"""
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (pred_flat * target_flat).sum()
        FP = ((1-target_flat) * pred_flat).sum()
        FN = (target_flat * (1-pred_flat)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        
        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss - Focuses on hard examples with small ROIs"""
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, smooth=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma
        
    def forward(self, pred, target):
        tversky_loss = self.tversky(pred, target)
        focal_tversky = torch.pow(tversky_loss, self.gamma)
        return focal_tversky


class SensitivitySpecificityLoss(nn.Module):
    """Sensitivity-Specificity Loss - For controlling TP/TN trade-off"""
    def __init__(self, w=0.5, smooth=1e-7):
        super(SensitivitySpecificityLoss, self).__init__()
        self.w = w
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True Positives, True Negatives, False Positives, False Negatives
        TP = (pred_flat * target_flat).sum()
        TN = ((1-pred_flat) * (1-target_flat)).sum()
        FP = ((1-target_flat) * pred_flat).sum()
        FN = (target_flat * (1-pred_flat)).sum()
        
        sensitivity = TP / (TP + FN + self.smooth)
        specificity = TN / (TN + FP + self.smooth)
        
        # Note: We return 1 - (weighted sum) to make it a loss
        return 1 - (self.w * sensitivity + (1 - self.w) * specificity)


class LogCoshDiceLoss(nn.Module):
    """Log-Cosh Dice Loss - Smooth version of Dice for better optimization"""
    def __init__(self, smooth=1.0):
        super(LogCoshDiceLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        
    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        return torch.log(torch.cosh(dice_loss))


class ComboLoss(nn.Module):
    """Combination of Dice and BCE"""
    def __init__(self, alpha=0.5, smooth=1.0):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss(smooth=smooth)
        
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        dice = self.dice_loss(pred, target)
        return self.alpha * bce + (1 - self.alpha) * dice


class ExponentialLogarithmicLoss(nn.Module):
    """Exponential Logarithmic Loss - Focuses on less accurately predicted structures"""
    def __init__(self, w_dice=0.5, w_cross=0.5, gamma_dice=0.3, gamma_cross=0.3):
        super(ExponentialLogarithmicLoss, self).__init__()
        self.w_dice = w_dice
        self.w_cross = w_cross
        self.gamma_dice = gamma_dice
        self.gamma_cross = gamma_cross
        self.dice_loss = DiceLoss()
        
    def forward(self, pred, target):
        # Dice component
        dice = self.dice_loss(pred, target)
        exp_dice = torch.mean(torch.pow(-torch.log(torch.clamp(1 - dice, min=1e-7)), self.gamma_dice))
        
        # Cross entropy component  
        pred_sigmoid = torch.sigmoid(pred)
        cross_entropy = F.binary_cross_entropy(pred_sigmoid, target, reduction='none')
        exp_cross = torch.mean(torch.pow(-torch.log(torch.clamp(1 - cross_entropy, min=1e-7)), self.gamma_cross))
        
        return self.w_dice * exp_dice + self.w_cross * exp_cross


class DistanceMapPenalizedCrossEntropy(nn.Module):
    """Distance map derived loss penalty term - For hard-to-segment boundaries"""
    def __init__(self, alpha=1.0):
        super(DistanceMapPenalizedCrossEntropy, self).__init__()
        self.alpha = alpha
        
    def forward(self, pred, target):
        # Generate distance map from ground truth
        dist_map = self._compute_distance_map(target)
        
        # Weight map: 1 + alpha * distance_map
        weight_map = 1 + self.alpha * dist_map
        
        # Weighted BCE
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = (weight_map * bce).mean()
        
        return weighted_bce
    
    def _compute_distance_map(self, target):
        """Compute distance transform for each sample in batch"""
        dist_maps = []
        target_np = target.cpu().numpy()
        
        for i in range(target_np.shape[0]):
            # For each channel
            channels = []
            for j in range(target_np.shape[1]):
                # Compute distance transform
                dist = distance_transform_edt(target_np[i, j])
                dist = dist / (dist.max() + 1e-7)  # Normalize
                channels.append(dist)
            dist_maps.append(np.stack(channels))
        
        dist_tensor = torch.from_numpy(np.stack(dist_maps)).float().to(target.device)
        return dist_tensor


class HausdorffDistanceLoss(nn.Module):
    """Approximated Hausdorff Distance Loss - For boundary-aware segmentation"""
    def __init__(self, alpha=2.0, reduction='mean'):
        super(HausdorffDistanceLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        
        # Compute distance transforms
        pred_dist = self._compute_dtm(pred_sigmoid)
        target_dist = self._compute_dtm(target)
        
        # Hausdorff distance approximation
        pred_error = (pred_sigmoid - target) ** 2
        distance = pred_dist ** self.alpha + target_dist ** self.alpha
        
        hd_loss = pred_error * distance
        
        if self.reduction == 'mean':
            return hd_loss.mean()
        elif self.reduction == 'sum':
            return hd_loss.sum()
        else:
            return hd_loss
    
    def _compute_dtm(self, img):
        """Compute distance transform map"""
        field = torch.zeros_like(img)
        
        for b in range(img.shape[0]):
            for c in range(img.shape[1]):
                img_np = img[b, c].cpu().numpy()
                if img_np.max() > 0:
                    dist = distance_transform_edt(img_np)
                    field[b, c] = torch.from_numpy(dist).float().to(img.device)
        
        return field / (field.max() + 1e-7)


class BoundaryLoss(nn.Module):
    """Boundary Loss - Emphasizes boundaries using distance maps"""
    def __init__(self, theta0=3, theta=5):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
        
    def forward(self, pred, target):
        # Generate boundary weight map
        with torch.no_grad():
            # Apply dilation and erosion to get boundaries
            dilated = F.max_pool2d(target, kernel_size=self.theta0, stride=1, padding=self.theta0//2)
            eroded = -F.max_pool2d(-target, kernel_size=self.theta0, stride=1, padding=self.theta0//2)
            boundary = (dilated - eroded)
            
            # Apply gaussian filter to boundary
            kernel_size = self.theta
            sigma = kernel_size / 3.0
            kernel = self._gaussian_kernel(kernel_size, sigma).to(target.device)
            kernel = kernel.view(1, 1, kernel_size, kernel_size)
            kernel = kernel.repeat(target.size(1), 1, 1, 1)
            
            boundary = F.conv2d(boundary, kernel, padding=kernel_size//2, groups=target.size(1))
            
            # Create weight map
            weight_map = 1 + 10 * boundary
        
        # Use weight map in weighted BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = (weight_map * bce).mean()
        
        return weighted_bce
    
    def _gaussian_kernel(self, size, sigma):
        """Create 2D Gaussian kernel"""
        coords = torch.arange(size).float()
        coords -= (size - 1) / 2.0
        
        g = coords**2
        g = (-(g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma**2)).exp()
        g /= g.sum()
        
        return g


# Additional compound loss functions
class TripleComboLoss(nn.Module):
    """Combination of Dice, BCE, and Focal Loss"""
    def __init__(self, alpha_dice=0.33, alpha_bce=0.33, alpha_focal=0.34, 
                 focal_alpha=0.25, gamma=2.0, smooth=1.0):
        super(TripleComboLoss, self).__init__()
        self.alpha_dice = alpha_dice
        self.alpha_bce = alpha_bce
        self.alpha_focal = alpha_focal
        
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = BCELoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=gamma)
        
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        return (self.alpha_dice * dice + 
                self.alpha_bce * bce + 
                self.alpha_focal * focal)


def get_loss_function(config):
    """Initialize the appropriate loss function based on config"""
    loss_name = config.get('loss_fn', 'combo')
    
    loss_functions = {
        'dice': lambda: DiceLoss(smooth=config.get('smooth', 1.0)),
        'bce': lambda: BCELoss(),
        'weighted_bce': lambda: WeightedBCELoss(beta=config.get('beta', 2.0)),
        'balanced_bce': lambda: BalancedBCELoss(),
        'focal': lambda: FocalLoss(
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0)
        ),
        'combo': lambda: ComboLoss(alpha=config.get('loss_alpha', 0.5)),
        'triple_combo': lambda: TripleComboLoss(
            alpha_dice=config.get('alpha_dice', 0.33),
            alpha_bce=config.get('alpha_bce', 0.33),
            alpha_focal=config.get('alpha_focal', 0.34),
            focal_alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0)
        ),
        'tversky': lambda: TverskyLoss(
            alpha=config.get('tversky_alpha', 0.5),
            beta=config.get('tversky_beta', 0.5)
        ),
        'tversky_balanced': lambda: TverskyLoss(alpha=0.5, beta=0.5),
        'tversky_recall': lambda: TverskyLoss(alpha=0.3, beta=0.7),
        'focal_tversky': lambda: FocalTverskyLoss(
            alpha=config.get('tversky_alpha', 0.5),
            beta=config.get('tversky_beta', 0.5),
            gamma=config.get('focal_tversky_gamma', 0.75)
        ),
        'sensitivity_specificity': lambda: SensitivitySpecificityLoss(
            w=config.get('sensitivity_weight', 0.5)
        ),
        'log_cosh_dice': lambda: LogCoshDiceLoss(smooth=config.get('smooth', 1.0)),
        'exponential_logarithmic': lambda: ExponentialLogarithmicLoss(
            w_dice=config.get('w_dice', 0.5),
            w_cross=config.get('w_cross', 0.5),
            gamma_dice=config.get('gamma_dice', 0.3),
            gamma_cross=config.get('gamma_cross', 0.3)
        ),
        'distance_map_bce': lambda: DistanceMapPenalizedCrossEntropy(
            alpha=config.get('distance_alpha', 1.0)
        ),
        'hausdorff': lambda: HausdorffDistanceLoss(
            alpha=config.get('hausdorff_alpha', 2.0)
        ),
        'boundary': lambda: BoundaryLoss(
            theta0=config.get('boundary_theta0', 3),
            theta=config.get('boundary_theta', 5)
        ),
    }
    
    if loss_name in loss_functions:
        return loss_functions[loss_name]()
    else:
        print(f"Warning: Unknown loss function '{loss_name}', defaulting to ComboLoss")
        return ComboLoss(alpha=config.get('loss_alpha', 0.5))


# Summary of loss functions and their use cases
LOSS_FUNCTION_GUIDE = """
Loss Function Selection Guide:
=============================

1. Binary Cross-Entropy (BCE):
   - Use for: Balanced datasets, general purpose
   - Pros: Stable training, well-understood
   - Cons: Poor with class imbalance

2. Weighted BCE:
   - Use for: Skewed datasets (more background than foreground)
   - Pros: Handles mild imbalance
   - Cons: Requires tuning beta parameter

3. Balanced BCE:
   - Use for: Automatically handles class imbalance
   - Pros: No manual weight tuning needed
   - Cons: May not work for extreme imbalance

4. Focal Loss:
   - Use for: Highly imbalanced datasets, hard examples
   - Pros: Focuses on hard-to-classify pixels
   - Cons: More hyperparameters to tune

5. Dice Loss:
   - Use for: Segmentation with class imbalance
   - Pros: Directly optimizes IoU-like metric
   - Cons: Can be unstable with very small objects

6. Tversky Loss:
   - Use for: When you need to control FP/FN trade-off
   - Pros: Flexible with alpha/beta parameters
   - Cons: Requires careful tuning

7. Focal Tversky Loss:
   - Use for: Small ROIs with high imbalance
   - Pros: Combines benefits of Focal and Tversky
   - Cons: Multiple hyperparameters

8. Sensitivity-Specificity Loss:
   - Use for: Medical imaging where TP/TN balance matters
   - Pros: Direct control over sensitivity/specificity
   - Cons: May not optimize overall accuracy

9. Log-Cosh Dice Loss:
   - Use for: Smooth optimization of Dice coefficient
   - Pros: More stable gradient flow
   - Cons: Slightly different from pure Dice optimization

10. Combo Loss (Dice + BCE):
    - Use for: General purpose, balanced approach
    - Pros: Combines benefits of both losses
    - Cons: Additional hyperparameter (alpha)

11. Exponential Logarithmic Loss:
    - Use for: Focusing on poorly predicted regions
    - Pros: Adaptive to prediction quality
    - Cons: Complex, multiple hyperparameters

12. Distance Map BCE:
    - Use for: Emphasizing boundaries
    - Pros: Better boundary delineation
    - Cons: Computational overhead

13. Hausdorff Distance Loss:
    - Use for: When boundary accuracy is critical
    - Pros: Directly optimizes boundary metric
    - Cons: Computationally expensive

14. Boundary Loss:
    - Use for: Precise boundary segmentation
    - Pros: Strong boundary emphasis
    - Cons: May ignore region interiors
"""

if __name__ == "__main__":
    print("Available loss functions:")
    print("========================")
    losses = [
        'dice', 'bce', 'weighted_bce', 'balanced_bce', 'focal',
        'tversky', 'focal_tversky', 'sensitivity_specificity',
        'log_cosh_dice', 'combo', 'triple_combo', 'exponential_logarithmic',
        'distance_map_bce', 'hausdorff', 'boundary'
    ]
    for loss in losses:
        print(f"  - {loss}")
    print("\nUse get_loss_function(config) to initialize a loss function")
    print("See LOSS_FUNCTION_GUIDE for selection guidance")
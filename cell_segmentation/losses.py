import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
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

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1.0):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss(smooth=smooth)
        
    def forward(self, pred, target):
        bce = nn.BCEWithLogitsLoss()(pred, target)
        dice = self.dice_loss(pred, target)
        return self.alpha * bce + (1 - self.alpha) * dice

class FocalLoss(nn.Module):
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

class BoundaryLoss(nn.Module):
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
            boundary = F.conv2d(boundary, 
                             torch.ones(1, 1, self.theta, self.theta).to(target.device) / (self.theta**2), 
                             padding=self.theta//2, groups=target.size(1))
            
            # Create weight map
            weight_map = 1 + 10 * boundary
        
        # Use weight map in weighted BCE loss
        bce = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none')
        weighted_bce = (weight_map * bce).mean()
        
        return weighted_bce

def get_loss_function(config):
    """Initialize the appropriate loss function based on config"""
    if config['loss_fn'] == 'combo':
        return ComboLoss(alpha=config.get('loss_alpha', 0.5))
    elif config['loss_fn'] == 'focal':
        return FocalLoss(alpha=config.get('loss_alpha', 0.25), 
                        gamma=config.get('focal_gamma', 2.0))
    elif config['loss_fn'] == 'tversky':
        return TverskyLoss(alpha=config.get('loss_alpha', 0.5),
                          beta=config.get('loss_beta', 0.5))
    elif config['loss_fn'] == 'boundary':
        return BoundaryLoss(theta0=config.get('boundary_theta0', 3),
                           theta=config.get('boundary_theta', 5))
    elif config['loss_fn'] == 'dice':
        return DiceLoss(smooth=config.get('smooth', 1.0))
    else:
        return ComboLoss(alpha=config.get('loss_alpha', 0.5))
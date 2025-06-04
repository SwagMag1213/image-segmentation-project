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

class BCELoss(nn.Module):
    """Standalone BCE Loss"""
    def __init__(self):
        super(BCELoss, self).__init__()
        
    def forward(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)

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

class FocalDiceComboLoss(nn.Module):
    def __init__(self, alpha=0.5, focal_alpha=0.25, gamma=2.0, smooth=1.0):
        super(FocalDiceComboLoss, self).__init__()
        self.alpha = alpha
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=gamma, reduction='mean')
        self.dice_loss = DiceLoss(smooth=smooth)
    
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.alpha * focal + (1 - self.alpha) * dice

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

class SoftNCutLoss(nn.Module):
    def __init__(self, img_size=(256, 256), radius=5, ox=4, oi=10):
        super().__init__()
        self.img_size = img_size
        self.radius = radius
        self.ox = ox
        self.oi = oi

    def forward(self, image, enc):
        return self.soft_n_cut_loss(image, enc)

    def soft_n_cut_loss(self, image, enc):
        loss = []
        batch_size = image.shape[0]
        k = enc.shape[1]
        weights = self.calculate_weights(image, batch_size)
        for i in range(k):
            loss.append(self.soft_n_cut_loss_single_k(weights, enc[:, i:i+1], batch_size))
        da = torch.stack(loss)
        return torch.mean(k - torch.sum(da, dim=0))

    def calculate_weights(self, input, batch_size):
        p = self.radius
        image = torch.mean(input, dim=1, keepdim=True)
        image = F.pad(image, (p, p, p, p), mode='constant', value=0)

        kh, kw = 2 * p + 1, 2 * p + 1
        patches = image.unfold(2, kh, 1).unfold(3, kw, 1)
        patches = patches.contiguous().view(batch_size, 1, -1, kh, kw)
        patches = patches.permute(0, 2, 1, 3, 4).view(-1, 1, kh, kw)

        center_values = patches[:, :, p, p].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, kh, kw)
        distance_weights = (torch.arange(kh) - p).view(1, -1).repeat(kh, 1).to(input.device)
        distance_weights = (distance_weights ** 2 + distance_weights.T ** 2).float()
        mask = distance_weights.le(self.radius)
        distance_weights = torch.exp(-distance_weights / self.ox ** 2) * mask

        patches = torch.exp(-((patches - center_values) ** 2) / self.oi ** 2)
        return patches * distance_weights

    def soft_n_cut_loss_single_k(self, weights, enc, batch_size):
        p = self.radius
        kh, kw = 2 * p + 1, 2 * p + 1
        encoding = F.pad(enc, (p, p, p, p), mode='constant', value=0)

        seg = encoding.unfold(2, kh, 1).unfold(3, kw, 1)
        seg = seg.contiguous().view(batch_size, 1, -1, kh, kw)
        seg = seg.permute(0, 2, 1, 3, 4).view(-1, 1, kh, kw)

        nom = weights * seg
        h, w = self.img_size
        numerator = torch.sum(enc * torch.sum(nom, dim=(1, 2, 3)).reshape(batch_size, h, w), dim=(1, 2, 3))
        denominator = torch.sum(enc * torch.sum(weights, dim=(1, 2, 3)).reshape(batch_size, h, w), dim=(1, 2, 3))

        return numerator / (denominator + 1e-8)

def get_loss_function(config):
    """Initialize the appropriate loss function based on config"""
    loss_name = config['loss_fn']
    
    if loss_name == 'dice':
        return DiceLoss(smooth=config.get('smooth', 1.0))
    elif loss_name == 'bce':
        return BCELoss()
    elif loss_name == 'combo':
        return ComboLoss(alpha=config.get('loss_alpha', 0.5))
    elif loss_name == 'focal':
        return FocalLoss(alpha=config.get('focal_alpha', 0.25), 
                        gamma=config.get('focal_gamma', 2.0))
    elif loss_name == 'triple_combo':
        return TripleComboLoss(
            alpha_dice=config.get('alpha_dice', 0.33),
            alpha_bce=config.get('alpha_bce', 0.33),
            alpha_focal=config.get('alpha_focal', 0.34),
            focal_alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0)
        )
    elif loss_name == 'tversky':
        return TverskyLoss(alpha=config.get('tversky_alpha', 0.5),
                          beta=config.get('tversky_beta', 0.5))
    elif loss_name == 'tversky_balanced':
        return TverskyLoss(alpha=0.5, beta=0.5)  # Balanced version
    elif loss_name == 'tversky_recall':
        return TverskyLoss(alpha=0.3, beta=0.7)  # Favor recall
    elif loss_name == 'boundary':
        return BoundaryLoss(theta0=config.get('boundary_theta0', 3),
                           theta=config.get('boundary_theta', 5))
    elif loss_name == 'softncut':
        return SoftNCutLoss(img_size=config.get('img_size', (256, 256)))
    else:
        # Default to ComboLoss
        return ComboLoss(alpha=config.get('loss_alpha', 0.5))
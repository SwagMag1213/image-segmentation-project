import torch
import torch.nn as nn
from advanced_models import UNetWithBackbone
import torch.nn.functional as F

class WNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, backbone='resnet34', pretrained=True, use_attention=True):
        super(WNet, self).__init__()
        self.encoder_unet = UNetWithBackbone(
            n_classes=n_classes,
            backbone=backbone,
            pretrained=pretrained,
            use_attention=use_attention
        )
        # For instance, if using both x2 (64 channels) and x3 (128 channels) for ResNet34:
        if backbone == 'resnet34':
            intermediate_channels = 64 + 128  # =192 channels
        elif backbone == 'resnet50':
            intermediate_channels = 512  # or a combination if desired
        elif backbone == 'densenet121':
            intermediate_channels = 512  # adjust as needed
        else:
            intermediate_channels = 128
        
        new_in_channels = n_classes + intermediate_channels
        self.decoder_unet = UNetWithBackbone(
            n_classes=n_channels,
            backbone=backbone,
            pretrained=pretrained,
            use_attention=use_attention
        )
        self.decoder_unet.input_conv = nn.Conv2d(new_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    def forward(self, x):
        # For example, returning x2 and x3:
        seg_output, features = self.encoder_unet(x, return_features=True)
        # If 'features' is the concatenation of x2 and x3
        features_up = F.interpolate(features, size=seg_output.shape[2:], mode='bilinear', align_corners=False)
        recon_input = torch.cat([seg_output, features_up], dim=1)
        recon_output = self.decoder_unet(recon_input)
        return seg_output, recon_output
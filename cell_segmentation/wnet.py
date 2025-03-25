import torch
import torch.nn as nn
from advanced_models import UNetWithBackbone

class WNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, backbone='resnet34', pretrained=True, use_attention=True):
        super(WNet, self).__init__()

        # Encoder U-Net for segmentation
        self.encoder_unet = UNetWithBackbone(
            n_classes=n_classes,
            backbone=backbone,
            pretrained=pretrained,
            use_attention=use_attention
        )

        # Decoder U-Net for reconstruction
        self.decoder_unet = UNetWithBackbone(
            n_classes=n_channels,
            backbone=backbone,
            pretrained=pretrained,
            use_attention=use_attention
        )

    def forward(self, x):
        # First U-Net produces segmentation mask
        seg_output = self.encoder_unet(x)

        # Second U-Net attempts to reconstruct the original image from the segmentation
        recon_output = self.decoder_unet(seg_output)

        return seg_output, recon_output

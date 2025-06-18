import torch
import torch.nn as nn
from torchvision import models

class UNetWithResnetEncoder(nn.Module):
    def __init__(self, n_classes=1):
        super(UNetWithResnetEncoder, self).__init__()
        
        # Load a pre-trained ResNet34 as the encoder
        resnet = models.resnet34(weights='IMAGENET1K_V1')
        
        # Modify first layer to accept grayscale input
        self.input_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.input_conv.weight.data = torch.sum(resnet.conv1.weight.data, dim=1, keepdim=True)
        
        # Encoder layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        # Decoder layers
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = self._decoder_block(512, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self._decoder_block(256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self._decoder_block(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = self._decoder_block(96, 32)  # 96 = 64 + 32
        
        # Final layer
        self.upconv0 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_final = nn.Conv2d(16, n_classes, kernel_size=1)
        
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder path
        x1 = self.input_conv(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)        # size: (512, 495)
        
        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)      # size: (256, 248)
        
        x3 = self.layer2(x2)      # size: (128, 124)
        x4 = self.layer3(x3)      # size: (64, 62)
        x5 = self.layer4(x4)      # size: (32, 31)
        
        # Decoder path with skip connections
        d4 = self.upconv4(x5)     # size: (64, 62)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)     # size: (128, 124)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)     # size: (256, 248)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)     # size: (512, 496)
        # Crop x1 if sizes don't match
        if d1.size() != x1.size():
            # Calculate center crop dimensions
            diff_h = x1.size(2) - d1.size(2)
            diff_w = x1.size(3) - d1.size(3)
            
            x1 = x1[:, :, 
                    diff_h//2:(diff_h//2 + d1.size(2)), 
                    diff_w//2:(diff_w//2 + d1.size(3))]
        
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.decoder1(d1)
        
        d0 = self.upconv0(d1)     # size: (1024, 992)
        # Crop to original size if needed
        if d0.size(2) != x.size(2) or d0.size(3) != x.size(3):
            # Calculate the padding or cropping needed
            diff_h = d0.size(2) - x.size(2)
            diff_w = d0.size(3) - x.size(3)
            
            if diff_h > 0 or diff_w > 0:
                # Need to crop
                d0 = d0[:, :, 
                       diff_h//2:(diff_h//2 + x.size(2)), 
                       diff_w//2:(diff_w//2 + x.size(3))]
        
        out = self.conv_final(d0)
        
        return out
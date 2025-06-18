import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Attention Gate module
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # Apply convolution operations
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Element-wise sum and ReLU
        psi = self.relu(g1 + x1)
        
        # Channel-wise attention map
        psi = self.psi(psi)
        
        # Element-wise multiplication
        return x * psi

# Channel Attention module (Squeeze-and-Excitation)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

# Base U-Net architecture with configurable backbone and attention
class UNetWithBackbone(nn.Module):
    def __init__(self, n_classes=1, backbone='resnet34', pretrained=True, use_attention=True):
        super(UNetWithBackbone, self).__init__()
        
        self.use_attention = use_attention
        self.backbone_name = backbone
        
        # Select the backbone
        if backbone == 'resnet34':
            backbone_model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            
            # Modify first layer to accept grayscale input
            self.input_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                self.input_conv.weight.data = torch.sum(backbone_model.conv1.weight.data, dim=1, keepdim=True)
            
            # Encoder layers
            self.bn1 = backbone_model.bn1
            self.relu = backbone_model.relu
            self.maxpool = backbone_model.maxpool
            self.enc1 = backbone_model.layer1  # 64 channels
            self.enc2 = backbone_model.layer2  # 128 channels
            self.enc3 = backbone_model.layer3  # 256 channels
            self.enc4 = backbone_model.layer4  # 512 channels
            
            # Decoder layers with correct channel dimensions
            self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.decoder4 = self._decoder_block(512, 256)  # 256 (skip) + 256 (up) = 512
            
            self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.decoder3 = self._decoder_block(256, 128)  # 128 (skip) + 128 (up) = 256
            
            self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.decoder2 = self._decoder_block(128, 64)   # 64 (skip) + 64 (up) = 128
            
            self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.decoder1 = self._decoder_block(96, 32)    # 64 (skip) + 32 (up) = 96
            
        elif backbone == 'resnet50':
            backbone_model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            
            # Modify first layer to accept grayscale input
            self.input_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                self.input_conv.weight.data = torch.sum(backbone_model.conv1.weight.data, dim=1, keepdim=True)
            
            # Encoder layers
            self.bn1 = backbone_model.bn1
            self.relu = backbone_model.relu
            self.maxpool = backbone_model.maxpool
            self.enc1 = backbone_model.layer1  # 256 channels
            self.enc2 = backbone_model.layer2  # 512 channels
            self.enc3 = backbone_model.layer3  # 1024 channels
            self.enc4 = backbone_model.layer4  # 2048 channels
            
            # Decoder layers with correct channel dimensions
            self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
            self.decoder4 = self._decoder_block(2048, 1024)  # 1024 (skip) + 1024 (up) = 2048
            
            self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.decoder3 = self._decoder_block(1024, 512)   # 512 (skip) + 512 (up) = 1024
            
            self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.decoder2 = self._decoder_block(512, 256)    # 256 (skip) + 256 (up) = 512
            
            self.upconv1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
            self.decoder1 = self._decoder_block(128, 64)     # 64 (skip) + 64 (up) = 128
            
        elif backbone == 'densenet121':
            backbone_model = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
            
            # Replace first layer with grayscale version
            self.input_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                self.input_conv.weight.data = torch.sum(
                    backbone_model.features.conv0.weight.data, dim=1, keepdim=True)
            
            self.features = backbone_model.features
            
            # DenseNet121 decoder layers
            # Feature map sizes: 64, 256, 512, 1024, 1024
            self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.decoder4 = self._decoder_block(1024, 512)  # 512 (skip) + 512 (up) = 1024
            
            self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.decoder3 = self._decoder_block(512, 256)   # 256 (skip) + 256 (up) = 512
            
            self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.decoder2 = self._decoder_block(256, 128)   # 128 (skip) + 128 (up) = 256
            
            self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.decoder1 = self._decoder_block(128, 64)    # 64 (skip) + 64 (up) = 128
            
        # Final layer
        self.upconv0 = nn.ConvTranspose2d(64 if backbone == 'densenet121' or backbone == 'resnet50' else 32, 
                                         16, kernel_size=2, stride=2)
        self.conv_final = nn.Conv2d(16, n_classes, kernel_size=1)
        
        # Configure attention gates if enabled
        if use_attention:
            if backbone == 'resnet34':
                self.attention4 = AttentionGate(F_g=256, F_l=256, F_int=128)
                self.attention3 = AttentionGate(F_g=128, F_l=128, F_int=64)
                self.attention2 = AttentionGate(F_g=64, F_l=64, F_int=32)
                self.attention1 = AttentionGate(F_g=32, F_l=64, F_int=32)
                
                self.ch_attention4 = ChannelAttention(256)
                self.ch_attention3 = ChannelAttention(128)
                self.ch_attention2 = ChannelAttention(64)
                self.ch_attention1 = ChannelAttention(32)
                
            elif backbone == 'resnet50':
                self.attention4 = AttentionGate(F_g=1024, F_l=1024, F_int=512)
                self.attention3 = AttentionGate(F_g=512, F_l=512, F_int=256)
                self.attention2 = AttentionGate(F_g=256, F_l=256, F_int=128)
                self.attention1 = AttentionGate(F_g=64, F_l=64, F_int=32)
                
                self.ch_attention4 = ChannelAttention(1024)
                self.ch_attention3 = ChannelAttention(512)
                self.ch_attention2 = ChannelAttention(256)
                self.ch_attention1 = ChannelAttention(64)
                
            elif backbone == 'densenet121':
                self.attention4 = AttentionGate(F_g=512, F_l=512, F_int=256)
                self.attention3 = AttentionGate(F_g=256, F_l=256, F_int=128)
                self.attention2 = AttentionGate(F_g=128, F_l=128, F_int=64)
                self.attention1 = AttentionGate(F_g=64, F_l=64, F_int=32)
                
                self.ch_attention4 = ChannelAttention(512)
                self.ch_attention3 = ChannelAttention(256)
                self.ch_attention2 = ChannelAttention(128)
                self.ch_attention1 = ChannelAttention(64)
        
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def _extract_densenet_features(self, x):
        """
        Properly extract 5 feature maps from DenseNet for U-Net skip connections
        """
        features = []
        
        # Process through initial convolution - this gives us the first feature map
        x1 = self.input_conv(x)
        
        # Need to access the internal DenseNet layers
        # First extract the individual components from the features sequential
        dense_layers = list(self.features.children())
        
        # Get norm, relu and pooling 
        norm0 = dense_layers[1]
        relu0 = dense_layers[2]
        pool0 = dense_layers[3]
        
        # Process through initial norm, relu, pool
        x = norm0(x1)
        x = relu0(x)
        x = pool0(x)
        
        # DenseBlock1
        denseblock1 = dense_layers[4]
        x = denseblock1(x)
        x2 = x  # Second feature map
        
        # Transition1
        transition1 = dense_layers[5]
        x = transition1(x)
        
        # DenseBlock2
        denseblock2 = dense_layers[6]
        x = denseblock2(x)
        x3 = x  # Third feature map
        
        # Transition2
        transition2 = dense_layers[7]
        x = transition2(x)
        
        # DenseBlock3
        denseblock3 = dense_layers[8]
        x = denseblock3(x)
        x4 = x  # Fourth feature map
        
        # Transition3
        transition3 = dense_layers[9]
        x = transition3(x)
        
        # DenseBlock4
        denseblock4 = dense_layers[10]
        x = denseblock4(x)
        x5 = x  # Fifth feature map
        
        return [x1, x2, x3, x4, x5]
        
    def forward(self, x, return_features=False):
        # Encoder path
        if self.backbone_name.startswith('resnet'):
            # ResNet encoder path
            x1 = self.input_conv(x)
            x1 = self.bn1(x1)
            x1 = self.relu(x1)
            
            x2 = self.maxpool(x1)
            x2 = self.enc1(x2)
            x3 = self.enc2(x2)
            x4 = self.enc3(x3)
            x5 = self.enc4(x4)
            
        elif self.backbone_name.startswith('densenet'):
            # DenseNet encoder path
            features = self._extract_densenet_features(x)
            x1, x2, x3, x4, x5 = features
            
        # Decoder path with skip connections
        d4 = self.upconv4(x5)
        if self.use_attention:
            # Apply attention gate
            x4_att = self.attention4(g=d4, x=x4)
            d4 = torch.cat((x4_att, d4), dim=1)
            d4 = self.decoder4(d4)
            d4 = self.ch_attention4(d4)
        else:
            d4 = torch.cat((x4, d4), dim=1)
            d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        if self.use_attention:
            x3_att = self.attention3(g=d3, x=x3)
            d3 = torch.cat((x3_att, d3), dim=1)
            d3 = self.decoder3(d3)
            d3 = self.ch_attention3(d3)
        else:
            d3 = torch.cat((x3, d3), dim=1)
            d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        if self.use_attention:
            x2_att = self.attention2(g=d2, x=x2)
            d2 = torch.cat((x2_att, d2), dim=1)
            d2 = self.decoder2(d2)
            d2 = self.ch_attention2(d2)
        else:
            d2 = torch.cat((x2, d2), dim=1)
            d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        
        # Handle size mismatches
        if d1.size() != x1.size():
            diff_h = x1.size(2) - d1.size(2)
            diff_w = x1.size(3) - d1.size(3)
            
            if diff_h > 0 and diff_w > 0:
                x1 = x1[:, :, 
                      diff_h//2:(diff_h//2 + d1.size(2)), 
                      diff_w//2:(diff_w//2 + d1.size(3))]
        
        if self.use_attention:
            x1_att = self.attention1(g=d1, x=x1)
            d1 = torch.cat((x1_att, d1), dim=1)
            d1 = self.decoder1(d1)
            d1 = self.ch_attention1(d1)
        else:
            d1 = torch.cat((x1, d1), dim=1)
            d1 = self.decoder1(d1)
        
        # Final upsampling
        d0 = self.upconv0(d1)
        
        # Handle final size mismatches
        if d0.size(2) != x.size(2) or d0.size(3) != x.size(3):
            diff_h = d0.size(2) - x.size(2)
            diff_w = d0.size(3) - x.size(3)
            
            if diff_h > 0 or diff_w > 0:
                d0 = d0[:, :, 
                      diff_h//2:(diff_h//2 + x.size(2)), 
                      diff_w//2:(diff_w//2 + x.size(3))]
        
        # Final convolution
        out = self.conv_final(d0)
        
        if return_features:
            # For example, fuse x2 and x3: upsample x2 to match x3
            x2_up = F.interpolate(x2, size=x3.shape[2:], mode='bilinear', align_corners=False)
            combined_features = torch.cat([x2_up, x3], dim=1)
            return out, combined_features
        return out
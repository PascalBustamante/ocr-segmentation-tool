import torch
import torch.nn as nn
import torch.nn.functional as F

class Segmentation(nn.Module):
    def __init__(self):
        super(Segmentation, self).__init__()
        #Encoder (downsampling)
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        # Decoder (upsampling)
        self.dec3 = self.conv_block(256, 128)
        self.dec2 = self.conv_block(128, 64)
        self.dec1 = self.conv_block(64, 32)
        
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        
        # Decoder
        d3 = self.dec3(F.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=True))
        d2 = self.dec2(F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True))
        d1 = self.dec1(d2)
        
        out = self.final(d1)
        return torch.sigmoid(out)
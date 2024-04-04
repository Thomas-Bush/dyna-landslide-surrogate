import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Complex_CNN(nn.Module):
    def __init__(self):
        super(Complex_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Input: 3 channels, Output: 32 channels
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 2, kernel_size=3, padding=1)  # Output: 2 channels (velocity and thickness)

    def forward(self, x):
        # Encoder path
        x1 = F.relu(self.bn1(self.conv1(x)))     # 64x64x32
        x2 = self.pool(x1)                       # 32x32x32
        x3 = F.relu(self.bn2(self.conv2(x2)))    # 32x32x64
        x4 = self.pool(x3)                       # 16x16x64
        x5 = F.relu(self.bn3(self.conv3(x4)))    # 16x16x128
        x6 = self.pool(x5)                       # 8x8x128
        x7 = F.relu(self.bn4(self.conv4(x6)))    # 8x8x256

        # Decoder path
        x8 = self.up(x7)                         # 16x16x256
        x9 = F.relu(self.bn5(self.conv5(x8)))    # 16x16x128
        x10 = self.up(x9)                        # 32x32x128
        x11 = F.relu(self.bn6(self.conv6(x10)))  # 32x32x64
        x12 = self.up(x11)                       # 64x64x64

        # Output layer
        x13 = self.conv7(x12)                    # 64x64x2
        return x13
    
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()

        # Define the U-Net architecture
        # Contracting Path (Encoder)
        self.enc_conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)

        # Expanding Path (Decoder)
        self.up_conv0 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.dec_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)  # Output: 1 channel (velocity)

    def forward(self, x):
        # Encoder
        enc0 = F.relu(self.enc_conv0(x))
        enc1 = F.relu(self.enc_conv1(self.pool(enc0)))
        enc2 = F.relu(self.enc_conv2(self.pool(enc1)))
        enc3 = F.relu(self.enc_conv3(self.pool(enc2)))

        # Decoder
        dec2 = F.relu(self.dec_conv1(torch.cat((self.up_conv0(enc3), enc2), dim=1)))
        dec1 = F.relu(self.dec_conv2(torch.cat((self.up_conv1(dec2), enc1), dim=1)))
        dec0 = F.relu(self.dec_conv3(torch.cat((self.up_conv2(dec1), enc0), dim=1)))

        # Final convolution
        return self.final_conv(dec0)
    
class LargeUNet(nn.Module):
    def __init__(self):
        super(LargeUNet, self).__init__()

        # Contracting Path (Encoder)
        self.enc_conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(64)
        self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.enc_conv4 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(1024)

        self.pool = nn.MaxPool2d(2, 2)

        # Expanding Path (Decoder)
        self.up_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv0 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.dec_conv4 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.dec_bn4 = nn.BatchNorm2d(512)
        self.dec_conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(256)
        self.dec_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(128)
        self.dec_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(64)

        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc0 = F.relu(self.bn0(self.enc_conv0(x)))
        enc1 = F.relu(self.bn1(self.enc_conv1(self.pool(enc0))))
        enc2 = F.relu(self.bn2(self.enc_conv2(self.pool(enc1))))
        enc3 = F.relu(self.bn3(self.enc_conv3(self.pool(enc2))))
        enc4 = F.relu(self.bn4(self.enc_conv4(self.pool(enc3))))

        # Decoder
        dec3 = F.relu(self.dec_bn4(self.dec_conv4(torch.cat((self.up_conv1(enc4), enc3), dim=1))))
        dec2 = F.relu(self.dec_bn3(self.dec_conv3(torch.cat((self.up_conv0(dec3), enc2), dim=1))))
        dec1 = F.relu(self.dec_bn2(self.dec_conv2(torch.cat((self.up_conv_1(dec2), enc1), dim=1))))
        dec0 = F.relu(self.dec_bn1(self.dec_conv1(torch.cat((self.up_conv_2(dec1), enc0), dim=1))))

        # Final convolution
        return self.final_conv(dec0)
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512, 1024]):
        super(UNet, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder path
        for feature in features:
            self.encoders.append(
                UNet._block(in_channels, feature)
            )
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = UNet._block(features[-1], features[-1] * 2)

        # Decoder path
        for feature in reversed(features):
            self.decoders.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(
                UNet._block(feature * 2, feature)
            )
        
        # Final convolution
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    @staticmethod
    def _block(in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip_connections = []

        # Encoder
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse the skip connections
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.decoders), 2):
            x = self.decoders[idx](x)
            skip_connection = skip_connections[idx // 2]

            # If the input sizes are different, resize the skip connection to match
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoders[idx + 1](x)

        return self.final_layer(x)
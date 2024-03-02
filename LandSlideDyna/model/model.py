import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py


class ConvBlock(nn.Module):
    """Convolutional Block consisting of two convolutional layers with Batch Normalization and ReLU activation."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DownBlock(nn.Module):
    """Downsampling Block with a MaxPool followed by a ConvBlock."""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """Upsampling Block with a ConvTranspose2d followed by a ConvBlock."""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpBlock, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutBlock(nn.Module):
    """Output Convolution Block to generate the final output."""
    def __init__(self, in_channels, out_channels):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

##########################
# TODO DOUBLE CHECK THIS #
##########################
    
class UNet(nn.Module):
    """U-Net architecture for feature extraction."""
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = ConvBlock(n_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.outc = OutBlock(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class CNNLSTMModel(nn.Module):
    """CNN-LSTM model for learning the sequence of landslide debris travel."""
    def __init__(self, in_channels, lstm_hidden_size, lstm_layers=1, n_classes=1):
        super(CNNLSTMModel, self).__init__()
        self.unet =UNet(n_channels=in_channels, n_classes=n_classes)
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)
        self.final_fc = nn.Linear(lstm_hidden_size, n_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        # Flatten the timesteps into the batch dimension
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.unet(c_in)
        # Reconstruct the temporal dimension
        c_out = c_out.view(batch_size, timesteps, -1)

        # Pass features through LSTM
        lstm_out, (h_n, c_n) = self.lstm(c_out)
        # Take only the output of the last LSTM cell
        final_output = self.final_fc(lstm_out[:, -1])
        return final_output